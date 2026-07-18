"""
Test suite for the Multi-Agent Clinical Research Assistant.

Test groups:
  - Conversational detection    (TestConversationalDetection)
  - PubMed tool                 (TestPubmedTool)
  - openFDA tool                (TestOpenFdaTool)
  - ClinicalTrials.gov tool     (TestClinicalTrialsTool)
  - RAG tool                    (TestRagTool)
  - Tavily tool                 (TestTavilyTool)
  - Orchestrator routing        (TestOrchestratorRouting)
  - Specialist agents           (TestSpecialistAgents)
  - End-to-end pipeline         (TestEndToEndPipeline)

All external calls (Groq, NCBI, openFDA, ClinicalTrials.gov, RAG service,
Tavily) are mocked so tests run offline with no API keys required.
"""

import os
import requests
from unittest.mock import MagicMock, patch

# Set dummy keys before any agent module is imported (they read it at import time)
os.environ.setdefault("GROQ_API_KEY", "test-key")
os.environ.setdefault("TAVILY_API_KEY", "test-key")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _groq_response(content: str) -> MagicMock:
    """Build a fake Groq API response object."""
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


PUBMED_XML = b"""<?xml version="1.0"?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>12345</PMID>
      <Article>
        <ArticleTitle>Anticoagulation in Atrial Fibrillation</ArticleTitle>
        <Abstract>
          <AbstractText>Warfarin remains a first-line anticoagulant.</AbstractText>
        </Abstract>
        <Journal>
          <JournalIssue>
            <PubDate><Year>2023</Year><Month>May</Month></PubDate>
          </JournalIssue>
        </Journal>
      </Article>
    </MedlineCitation>
  </PubmedArticle>
</PubmedArticleSet>
"""


# ---------------------------------------------------------------------------
# Conversational detection
# ---------------------------------------------------------------------------

class TestConversationalDetection:

    @patch("agents.orchestrator._client")
    def test_greeting_is_conversational(self, mock_client):
        """A plain greeting should be handled directly without agents."""
        mock_client.chat.completions.create.return_value = _groq_response(
            '{"type": "conversational", "reply": "Hello! How can I help?"}'
        )
        from agents.orchestrator import _is_conversational
        is_conv, reply = _is_conversational("Hello!")
        assert is_conv is True
        assert reply != ""

    @patch("agents.orchestrator._client")
    def test_name_introduction_is_conversational(self, mock_client):
        """Introducing a name is not a clinical research query."""
        mock_client.chat.completions.create.return_value = _groq_response(
            '{"type": "conversational", "reply": "Nice to meet you, Yamini!"}'
        )
        from agents.orchestrator import _is_conversational
        is_conv, reply = _is_conversational("My name is Yamini.")
        assert is_conv is True
        assert reply != ""

    @patch("agents.orchestrator._client")
    def test_clinical_query_is_not_conversational(self, mock_client):
        """A clear clinical question should not be classified as conversational."""
        mock_client.chat.completions.create.return_value = _groq_response(
            '{"type": "research"}'
        )
        from agents.orchestrator import _is_conversational
        is_conv, _ = _is_conversational("What are treatment options for atrial fibrillation?")
        assert is_conv is False

    @patch("agents.orchestrator._client")
    def test_small_talk_is_conversational(self, mock_client):
        """Small talk like 'how are you' should not trigger agents."""
        mock_client.chat.completions.create.return_value = _groq_response(
            '{"type": "conversational", "reply": "I\'m doing great, thanks!"}'
        )
        from agents.orchestrator import _is_conversational
        is_conv, _ = _is_conversational("How are you doing today?")
        assert is_conv is True

    @patch("agents.orchestrator._client")
    def test_malformed_json_defaults_to_research(self, mock_client):
        """If Groq returns unparseable output, default to treating as research."""
        mock_client.chat.completions.create.return_value = _groq_response(
            "Sorry, I cannot classify this."
        )
        from agents.orchestrator import _is_conversational
        is_conv, _ = _is_conversational("What is warfarin used for?")
        assert is_conv is False


# ---------------------------------------------------------------------------
# PubMed tool
# ---------------------------------------------------------------------------

class TestPubmedTool:

    @patch("tools.pubmed_tool.requests.get")
    def test_returns_list_of_dicts(self, mock_get):
        """search_pubmed should return parsed article dicts from esearch + efetch."""
        search_resp = MagicMock()
        search_resp.raise_for_status.return_value = None
        search_resp.json.return_value = {"esearchresult": {"idlist": ["12345"]}}

        fetch_resp = MagicMock()
        fetch_resp.raise_for_status.return_value = None
        fetch_resp.content = PUBMED_XML

        mock_get.side_effect = [search_resp, fetch_resp]

        from tools.pubmed_tool import search_pubmed
        results = search_pubmed("atrial fibrillation")
        assert isinstance(results, list)
        assert results[0]["pmid"] == "12345"
        assert "Anticoagulation" in results[0]["title"]
        assert results[0]["published"] == "2023-May"

    @patch("tools.pubmed_tool.requests.get")
    def test_empty_idlist_returns_empty_list(self, mock_get):
        """If esearch finds no PMIDs, tool should not call efetch and return []."""
        search_resp = MagicMock()
        search_resp.raise_for_status.return_value = None
        search_resp.json.return_value = {"esearchresult": {"idlist": []}}
        mock_get.return_value = search_resp

        from tools.pubmed_tool import search_pubmed
        results = search_pubmed("xyznonexistenttopic12345")
        assert results == []
        assert mock_get.call_count == 1

    @patch("tools.pubmed_tool.time.sleep")
    @patch("tools.pubmed_tool.requests.get")
    def test_request_exception_returns_empty_list(self, mock_get, mock_sleep):
        """Network failures should degrade gracefully to an empty list, not crash."""
        mock_get.side_effect = requests.RequestException("connection refused")
        from tools.pubmed_tool import search_pubmed
        results = search_pubmed("atrial fibrillation")
        assert results == []

    @patch("tools.pubmed_tool.requests.get")
    def test_abstract_truncated_to_1000_chars(self, mock_get):
        """Untruncated abstracts across several articles can blow past a single
        Groq request's token budget — cap each one like the sibling tools do."""
        search_resp = MagicMock()
        search_resp.raise_for_status.return_value = None
        search_resp.json.return_value = {"esearchresult": {"idlist": ["99999"]}}

        long_abstract_xml = f"""<?xml version="1.0"?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>99999</PMID>
      <Article>
        <ArticleTitle>Long Abstract Paper</ArticleTitle>
        <Abstract>
          <AbstractText>{"A" * 2000}</AbstractText>
        </Abstract>
      </Article>
    </MedlineCitation>
  </PubmedArticle>
</PubmedArticleSet>
"""
        fetch_resp = MagicMock()
        fetch_resp.raise_for_status.return_value = None
        fetch_resp.content = long_abstract_xml.encode()
        mock_get.side_effect = [search_resp, fetch_resp]

        from tools.pubmed_tool import search_pubmed
        results = search_pubmed("test")
        assert len(results[0]["abstract"]) == 1000


# ---------------------------------------------------------------------------
# openFDA tool
# ---------------------------------------------------------------------------

class TestOpenFdaTool:

    @patch("tools.openfda_tool.requests.get")
    def test_returns_label_with_key_fields(self, mock_get):
        """search_drug_label should surface brand/generic name and label sections."""
        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status.return_value = None
        resp.json.return_value = {
            "results": [{
                "openfda": {
                    "brand_name": ["Coumadin"],
                    "generic_name": ["warfarin"],
                    "manufacturer_name": ["Bristol-Myers Squibb"],
                },
                "indications_and_usage": ["Used to prevent blood clots."],
                "warnings": ["May cause major bleeding."],
            }]
        }
        mock_get.return_value = resp

        from tools.openfda_tool import search_drug_label
        results = search_drug_label("warfarin")
        assert results[0]["brand_name"] == "Coumadin"
        assert results[0]["generic_name"] == "warfarin"
        assert "indications_and_usage" in results[0]
        assert "warnings" in results[0]

    @patch("tools.openfda_tool.requests.get")
    def test_404_returns_empty_list(self, mock_get):
        """openFDA returns 404 (not a normal error body) on zero matches."""
        resp = MagicMock()
        resp.status_code = 404
        mock_get.return_value = resp

        from tools.openfda_tool import search_drug_label
        results = search_drug_label("nonexistentdrugxyz")
        assert results == []

    @patch("tools.openfda_tool.time.sleep")
    @patch("tools.openfda_tool.requests.get")
    def test_request_exception_returns_empty_list(self, mock_get, mock_sleep):
        """Network failures should degrade gracefully to an empty list, not crash."""
        mock_get.side_effect = requests.RequestException("connection refused")
        from tools.openfda_tool import search_drug_label
        results = search_drug_label("warfarin")
        assert results == []

    @patch("tools.openfda_tool.requests.get")
    def test_label_field_truncated_to_1000_chars(self, mock_get):
        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status.return_value = None
        resp.json.return_value = {
            "results": [{"openfda": {}, "warnings": ["W" * 2000]}]
        }
        mock_get.return_value = resp

        from tools.openfda_tool import search_drug_label
        results = search_drug_label("test")
        assert len(results[0]["warnings"]) == 1000


# ---------------------------------------------------------------------------
# ClinicalTrials.gov tool
# ---------------------------------------------------------------------------

class TestClinicalTrialsTool:

    @patch("tools.clinicaltrials_tool.requests.get")
    def test_returns_parsed_studies(self, mock_get):
        """search_clinical_trials should flatten the nested v2 API response."""
        resp = MagicMock()
        resp.raise_for_status.return_value = None
        resp.json.return_value = {
            "studies": [{
                "protocolSection": {
                    "identificationModule": {"nctId": "NCT00000001", "briefTitle": "AFib Anticoagulation Study"},
                    "statusModule": {"overallStatus": "RECRUITING"},
                    "designModule": {"phases": ["PHASE3"]},
                    "contactsLocationsModule": {
                        "locations": [{"city": "Boston", "country": "United States"}]
                    },
                }
            }]
        }
        mock_get.return_value = resp

        from tools.clinicaltrials_tool import search_clinical_trials
        results = search_clinical_trials("atrial fibrillation")
        assert results[0]["nct_id"] == "NCT00000001"
        assert results[0]["status"] == "RECRUITING"
        assert "Boston, United States" in results[0]["locations"]

    @patch("tools.clinicaltrials_tool.requests.get")
    def test_no_studies_returns_empty_list(self, mock_get):
        resp = MagicMock()
        resp.raise_for_status.return_value = None
        resp.json.return_value = {"studies": []}
        mock_get.return_value = resp

        from tools.clinicaltrials_tool import search_clinical_trials
        results = search_clinical_trials("nonexistent condition xyz")
        assert results == []

    @patch("tools.clinicaltrials_tool.time.sleep")
    @patch("tools.clinicaltrials_tool.requests.get")
    def test_request_exception_returns_empty_list(self, mock_get, mock_sleep):
        """Network failures should degrade gracefully to an empty list, not crash."""
        mock_get.side_effect = requests.RequestException("connection refused")
        from tools.clinicaltrials_tool import search_clinical_trials
        results = search_clinical_trials("atrial fibrillation")
        assert results == []

    @patch("tools.clinicaltrials_tool.requests.get")
    def test_locations_truncated_to_five(self, mock_get):
        resp = MagicMock()
        resp.raise_for_status.return_value = None
        resp.json.return_value = {
            "studies": [{
                "protocolSection": {
                    "identificationModule": {"nctId": "NCT1", "briefTitle": "Multi-site Trial"},
                    "statusModule": {"overallStatus": "RECRUITING"},
                    "designModule": {"phases": ["PHASE2"]},
                    "contactsLocationsModule": {
                        "locations": [{"city": f"City{i}", "country": "US"} for i in range(8)]
                    },
                }
            }]
        }
        mock_get.return_value = resp

        from tools.clinicaltrials_tool import search_clinical_trials
        results = search_clinical_trials("test condition")
        assert len(results[0]["locations"]) == 5


# ---------------------------------------------------------------------------
# RAG tool
# ---------------------------------------------------------------------------

class TestRagTool:

    @patch("tools.rag_tool.requests.post")
    def test_returns_json_on_success(self, mock_post):
        resp = MagicMock()
        resp.raise_for_status.return_value = None
        resp.json.return_value = {
            "answer": "Anticoagulation is first-line therapy.",
            "sources": [{"title": "Atrial Fibrillation", "pmid": "123", "nbk_id": "NBK1"}],
        }
        mock_post.return_value = resp

        from tools.rag_tool import ask_rag
        result = ask_rag("What is atrial fibrillation treatment?")
        assert result["answer"] == "Anticoagulation is first-line therapy."
        assert result["sources"][0]["title"] == "Atrial Fibrillation"

    @patch("tools.rag_tool.requests.post")
    def test_service_down_returns_none(self, mock_post):
        """If the RAG service is unreachable, the tool should return None, not raise."""
        mock_post.side_effect = requests.RequestException("connection refused")
        from tools.rag_tool import ask_rag
        result = ask_rag("some clinical question")
        assert result is None


# ---------------------------------------------------------------------------
# Tavily tool
# ---------------------------------------------------------------------------

class TestTavilyTool:

    @patch("tools.tavily_tool.TavilyClient")
    def test_returns_list_of_results(self, mock_client_cls):
        mock_client_cls.return_value.search.return_value = {
            "results": [
                {"title": "New AFib Guidelines", "url": "http://example.com", "content": "Updated guidance."}
            ]
        }
        from tools.tavily_tool import search_web
        results = search_web("latest atrial fibrillation guidelines")
        assert isinstance(results, list)
        assert results[0]["title"] == "New AFib Guidelines"

    @patch("tools.tavily_tool.TavilyClient")
    def test_content_truncated_to_500_chars(self, mock_client_cls):
        mock_client_cls.return_value.search.return_value = {
            "results": [
                {"title": "Article", "url": "http://example.com", "content": "Z" * 1000}
            ]
        }
        from tools.tavily_tool import search_web
        results = search_web("test")
        assert len(results[0]["content"]) == 500

    @patch("tools.tavily_tool.TavilyClient")
    def test_empty_results_returns_empty_list(self, mock_client_cls):
        mock_client_cls.return_value.search.return_value = {"results": []}
        from tools.tavily_tool import search_web
        results = search_web("obscure query xyz")
        assert results == []

    @patch("tools.tavily_tool.TavilyClient")
    def test_api_error_returns_empty_list(self, mock_client_cls):
        """Tavily rejects some queries outright (e.g. HTTP 422) rather than
        just returning zero results — this must degrade gracefully too."""
        mock_client_cls.return_value.search.side_effect = requests.HTTPError("422 Client Error")
        from tools.tavily_tool import search_web
        results = search_web("a malformed or too-short query")
        assert results == []


# ---------------------------------------------------------------------------
# Orchestrator routing
# ---------------------------------------------------------------------------

class TestOrchestratorRouting:

    @patch("agents.orchestrator._client")
    def test_literature_query_routes_to_pubmed(self, mock_client):
        mock_client.chat.completions.create.return_value = _groq_response(
            '{"pubmed": "atrial fibrillation treatment"}'
        )
        from agents.orchestrator import _route_and_split
        routing = _route_and_split("What are treatment options for atrial fibrillation?")
        assert "pubmed" in routing

    @patch("agents.orchestrator._client")
    def test_drug_query_routes_to_openfda(self, mock_client):
        mock_client.chat.completions.create.return_value = _groq_response(
            '{"openfda": "warfarin"}'
        )
        from agents.orchestrator import _route_and_split
        routing = _route_and_split("What are the warnings for warfarin?")
        assert "openfda" in routing

    @patch("agents.orchestrator._client")
    def test_trial_query_routes_to_clinicaltrials(self, mock_client):
        mock_client.chat.completions.create.return_value = _groq_response(
            '{"clinicaltrials": "atrial fibrillation"}'
        )
        from agents.orchestrator import _route_and_split
        routing = _route_and_split("Are there active trials for atrial fibrillation?")
        assert "clinicaltrials" in routing

    @patch("agents.orchestrator._client")
    def test_current_events_routes_to_tavily(self, mock_client):
        mock_client.chat.completions.create.return_value = _groq_response(
            '{"tavily": "latest FDA drug recall news"}'
        )
        from agents.orchestrator import _route_and_split
        routing = _route_and_split("What is the latest drug recall news?")
        assert "tavily" in routing

    @patch("agents.orchestrator._client")
    def test_no_duplicate_agents_in_routing(self, mock_client):
        mock_client.chat.completions.create.return_value = _groq_response(
            '{"pubmed": "afib treatment", "openfda": "warfarin"}'
        )
        from agents.orchestrator import _route_and_split
        routing = _route_and_split("afib treatment and warfarin warnings")
        assert len(routing) == len(set(routing.keys()))

    @patch("agents.orchestrator._client")
    def test_invalid_json_falls_back_to_all_agents(self, mock_client):
        mock_client.chat.completions.create.return_value = _groq_response(
            "I cannot decide which agents to use."
        )
        from agents.orchestrator import _route_and_split, AGENT_REGISTRY
        routing = _route_and_split("some query")
        assert set(routing.keys()) == set(AGENT_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Specialist agents
# ---------------------------------------------------------------------------

class TestSpecialistAgents:

    @patch("agents.pubmed_agent.call_groq")
    @patch("agents.pubmed_agent.search_pubmed")
    def test_pubmed_agent_returns_string(self, mock_search, mock_call_groq):
        mock_search.return_value = [{"pmid": "123", "title": "AFib Study", "abstract": "...", "published": "2023-May"}]
        mock_call_groq.return_value = "Summary of AFib literature."
        from agents.pubmed_agent import run_pubmed_agent
        result = run_pubmed_agent("atrial fibrillation")
        assert isinstance(result, str)
        assert len(result) > 0

    @patch("agents.pubmed_agent.search_pubmed")
    def test_pubmed_agent_handles_empty_results(self, mock_search):
        mock_search.return_value = []
        from agents.pubmed_agent import run_pubmed_agent
        result = run_pubmed_agent("xyznonexistenttopic")
        assert "No PubMed articles" in result

    @patch("agents.openfda_agent.call_groq")
    @patch("agents.openfda_agent.search_drug_label")
    def test_openfda_agent_returns_string(self, mock_search, mock_call_groq):
        mock_search.return_value = [{"brand_name": "Coumadin", "generic_name": "warfarin", "warnings": "May cause bleeding."}]
        mock_call_groq.return_value = "Warfarin warnings summary."
        from agents.openfda_agent import run_openfda_agent
        result = run_openfda_agent("warfarin")
        assert isinstance(result, str)

    @patch("agents.openfda_agent.search_drug_label")
    def test_openfda_agent_handles_empty_results(self, mock_search):
        mock_search.return_value = []
        from agents.openfda_agent import run_openfda_agent
        result = run_openfda_agent("nonexistentdrugxyz")
        assert "No openFDA drug label" in result

    @patch("agents.clinicaltrials_agent.call_groq")
    @patch("agents.clinicaltrials_agent.search_clinical_trials")
    def test_clinicaltrials_agent_returns_string(self, mock_search, mock_call_groq):
        mock_search.return_value = [{
            "nct_id": "NCT001", "title": "AFib Trial", "status": "RECRUITING",
            "phases": ["PHASE3"], "locations": ["Boston, United States"],
        }]
        mock_call_groq.return_value = "Trial summary."
        from agents.clinicaltrials_agent import run_clinicaltrials_agent
        result = run_clinicaltrials_agent("atrial fibrillation")
        assert isinstance(result, str)

    @patch("agents.clinicaltrials_agent.search_clinical_trials")
    def test_clinicaltrials_agent_handles_empty_results(self, mock_search):
        mock_search.return_value = []
        from agents.clinicaltrials_agent import run_clinicaltrials_agent
        result = run_clinicaltrials_agent("nonexistent condition xyz")
        assert "No active ClinicalTrials.gov trials" in result

    @patch("agents.rag_agent.ask_rag")
    def test_rag_agent_formats_answer_with_sources(self, mock_ask):
        mock_ask.return_value = {
            "answer": "Anticoagulation is first-line therapy.",
            "sources": [{"title": "Atrial Fibrillation", "pmid": "123", "nbk_id": "NBK1"}],
        }
        from agents.rag_agent import run_rag_agent
        result = run_rag_agent("atrial fibrillation treatment")
        assert "Anticoagulation is first-line therapy." in result
        assert "Atrial Fibrillation" in result

    @patch("agents.rag_agent.ask_rag")
    def test_rag_agent_handles_service_down(self, mock_ask):
        mock_ask.return_value = None
        from agents.rag_agent import run_rag_agent
        result = run_rag_agent("some clinical question")
        assert "unavailable" in result

    @patch("agents.tavily_agent.call_groq")
    @patch("agents.tavily_agent.search_web")
    def test_tavily_agent_returns_string(self, mock_search, mock_call_groq):
        mock_search.return_value = [{"title": "Health News", "url": "http://news.com", "content": "Breaking health news..."}]
        mock_call_groq.return_value = "Here is today's health news."
        from agents.tavily_agent import run_tavily_agent
        result = run_tavily_agent("latest health news")
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# End-to-end pipeline
# ---------------------------------------------------------------------------

class TestEndToEndPipeline:

    @patch("agents.orchestrator._client")
    @patch("agents.pubmed_agent.call_groq")
    @patch("agents.pubmed_agent.search_pubmed")
    def test_clinical_query_returns_final_answer(
        self, mock_search, mock_pubmed_call_groq, mock_orch_client
    ):
        """A clinical query should go through the full pipeline and return a string answer."""
        mock_orch_client.chat.completions.create.side_effect = [
            _groq_response('{"type": "research"}'),
            _groq_response('{"pubmed": "atrial fibrillation treatment"}'),
            _groq_response("Anticoagulants are first-line therapy for atrial fibrillation."),
        ]
        mock_search.return_value = [{"pmid": "123", "title": "AFib Management", "abstract": "...", "published": "2023-May"}]
        mock_pubmed_call_groq.return_value = "AFib Management discusses anticoagulation strategies."

        from agents.orchestrator import run_orchestrator
        result = run_orchestrator("What are treatment options for atrial fibrillation?")
        assert isinstance(result, str)
        assert len(result) > 0

    @patch("agents.orchestrator._client")
    def test_conversational_query_skips_agents(self, mock_client):
        """A conversational query should return a direct reply — no agent is called."""
        mock_client.chat.completions.create.return_value = _groq_response(
            '{"type": "conversational", "reply": "Nice to meet you!"}'
        )

        with patch("agents.orchestrator.run_pubmed_agent") as mock_pubmed, \
             patch("agents.orchestrator.run_openfda_agent") as mock_fda, \
             patch("agents.orchestrator.run_clinicaltrials_agent") as mock_ct, \
             patch("agents.orchestrator.run_rag_agent") as mock_rag, \
             patch("agents.orchestrator.run_tavily_agent") as mock_tavily:

            from agents.orchestrator import run_orchestrator
            result = run_orchestrator("My name is Yamini.")

            mock_pubmed.assert_not_called()
            mock_fda.assert_not_called()
            mock_ct.assert_not_called()
            mock_rag.assert_not_called()
            mock_tavily.assert_not_called()
            assert isinstance(result, str)
