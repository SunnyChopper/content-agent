from typing import TypedDict, Annotated, List, Optional
from langgraph.graph.message import add_messages

from models.states.agents.editor_state import EditorInsights, EditorFeedback, EditorSuggestions, EditorSuggestionsApplied
from models.states.agents.researcher_state import ResearchHyperlink, FollowUpTopics, ResearchInsights
from models.states.agents.writer_state import BusinessGoal, SEOKeyword, ContentOutline, BlogPost

class BlogWorkflowState(TypedDict):
    # Base workflow state
    messages: Annotated[list, add_messages] = []
    
    # Researcher state
    research_insights: Optional[ResearchInsights] = None
    search_queries: List[str] = []
    new_search_queries: List[str] = []
    confidence_level: float = 0.0
    hyperlinks: List[ResearchHyperlink] = []
    content_query: str = ""
    current_loop: int = 0
    follow_up_topics: Optional[FollowUpTopics] = None
    
    # Writer state
    business_goals: List[BusinessGoal] = []
    target_read_time: Optional[int] = None
    seo_keywords: List[SEOKeyword] = []
    content_outline: Optional[ContentOutline] = None
    blog_post: Optional[BlogPost] = None
    current_writing_step: str = "generate_outline"

    # Editor state
    editor_insights: Optional[EditorInsights] = None
    editor_feedback: Optional[EditorFeedback] = None
    editor_suggestions: Optional[EditorSuggestions] = None
    editor_suggestions_applied: Optional[EditorSuggestionsApplied] = None

