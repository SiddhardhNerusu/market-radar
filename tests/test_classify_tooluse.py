"""Lock the forced tool-use classification path (ingestion blueprint #3 L137):
the model returns schema-valid JSON via a forced tool call (no json.loads reject
path), and low-confidence labels are bucketed away from the trade gate."""
from market_radar.llm.classifier import LLMClassifier


class _Usage:
    input_tokens = 10
    output_tokens = 20


class _ToolBlock:
    type = "tool_use"
    name = "classify_signal"

    def __init__(self, inp):
        self.input = inp


class _Msg:
    def __init__(self, content):
        self.content = content
        self.usage = _Usage()


class _Messages:
    def __init__(self, content):
        self._content = content
        self.last_kwargs = None

    def create(self, **kw):
        self.last_kwargs = kw
        return _Msg(self._content)


class _Client:
    def __init__(self, content):
        self.messages = _Messages(content)


class _Tracker:
    daily_cap_usd = 10.0

    def can_spend(self, est):
        return (True, 0.0, 10.0)

    def record(self, cost_usd=0.0):
        pass


def _classifier(content):
    c = LLMClassifier.__new__(LLMClassifier)
    c.model, c.max_retries, c.retry_backoff = "m", 1, 1.0
    c._client = _Client(content)
    c.tracker = _Tracker()
    return c


_ROW = {"title": "t", "body": "b", "signal_id": 1, "ticker": "X"}


def test_forced_tool_is_requested_and_parsed():
    c = _classifier([_ToolBlock({"event_type": "earnings_beat", "sentiment": 0.6,
                                 "sentiment_magnitude": 0.7, "factual": 1,
                                 "tickers_mentioned": ["X"], "extracted_fields": {},
                                 "confidence": 0.9})])
    r = c.classify_one(_ROW)
    assert r and r["event_type"] == "earnings_beat"
    kw = c._client.messages.last_kwargs
    assert kw["tools"] and kw["tool_choice"]["name"] == "classify_signal"


def test_low_confidence_is_bucketed():
    c = _classifier([_ToolBlock({"event_type": "m_a_announcement", "sentiment": 0.2,
                                 "sentiment_magnitude": 0.3, "factual": 0,
                                 "tickers_mentioned": [], "extracted_fields": {},
                                 "confidence": 0.3})])
    r = c.classify_one(_ROW)
    assert r and r["event_type"] == "unclassified_low_confidence"


def test_no_tool_call_returns_none():
    class _Text:
        type = "text"
        text = "I refuse"
    r = _classifier([_Text()]).classify_one(_ROW)
    assert r is None
