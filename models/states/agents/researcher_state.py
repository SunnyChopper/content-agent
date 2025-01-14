from typing_extensions import TypedDict
from typing import List, Optional
from pydantic import BaseModel

# --- Langgraph Types --- #
class ResearchHyperlink(TypedDict):
    link: str
    title: str
    snippet: str

    def __add__(self, other: 'ResearchHyperlink') -> 'ResearchHyperlink':
        return ResearchHyperlink(link=self.link + other.link, title=self.title + other.title, snippet=self.snippet + other.snippet)

class SearchQueries(BaseModel):
    generated_queries: List[str]

class FollowUpTopics(BaseModel):
    topics: List[str]

    def __len__(self) -> int:
        return len(self.topics)

class InsightPoint(BaseModel):
    main_point: str
    supporting_points: List[str]

class ResearchInsights(BaseModel):
    insights: List[InsightPoint]
    hyperlinks: List[ResearchHyperlink]
    new_search_queries: Optional[List[str]] = None

    def __add__(self, other: 'ResearchInsights') -> 'ResearchInsights':
        return ResearchInsights(
            insights=self.insights + (other.insights or []),
            hyperlinks=self.hyperlinks + (other.hyperlinks or []),
            new_search_queries=(self.new_search_queries or []) + (other.new_search_queries or [])
        )
    
    def __str__(self) -> str:
        return f"Insights:\n{self.insights}\n\nHyperlinks:\n{self.hyperlinks}\n\nNew Search Queries:\n{self.new_search_queries}\n\n"

class DimensionScores(BaseModel):
    comprehensiveness: float
    evidence_quality: float
    clarity: float
    practical_application: float
    coherence: float

class DimensionFeedback(BaseModel):
    comprehensiveness: str
    evidence_quality: str
    clarity: str
    practical_application: str
    coherence: str

class ConfidenceLevel(BaseModel):
    dimension_scores: DimensionScores
    dimension_feedback: DimensionFeedback
    confidence_score: float

class ResearchHyperlinks(BaseModel):
    hyperlinks: List[ResearchHyperlink]

class SearchResult(TypedDict):
    title: str
    link: str
    snippet: str
    full_content: Optional[str] = None

class SearchResults(BaseModel):
    results: List[SearchResult]

class ResearcherState(TypedDict):
    research_insights: ResearchInsights
    search_queries: List[str]
    new_search_queries: List[str]
    confidence_level: float
    hyperlinks: List[ResearchHyperlink]
    content_query: str
    current_loop: int
    follow_up_topics: FollowUpTopics