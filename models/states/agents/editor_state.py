from typing import TypedDict, List, Optional

class EditorInsights(TypedDict):
    content_quality: float
    readability_score: float
    seo_alignment: float
    insights: List[str]

class EditorFeedback(TypedDict):
    strengths: List[str] 
    weaknesses: List[str]
    improvement_areas: List[str]

class EditorSuggestions(TypedDict):
    content_suggestions: List[str]
    style_suggestions: List[str]
    structure_suggestions: List[str]

class EditorSuggestionsApplied(TypedDict):
    applied_suggestions: List[str]
    skipped_suggestions: List[str]
    rationale: str

class EditorState(TypedDict):
    editor_insights: Optional[EditorInsights]
    editor_feedback: Optional[EditorFeedback] 
    editor_suggestions: Optional[EditorSuggestions]
    editor_suggestions_applied: Optional[EditorSuggestionsApplied]
    final_content: str
