from typing import List, Optional
from typing_extensions import TypedDict
from pydantic import BaseModel

class SEOKeyword(BaseModel):
    keyword: str
    search_volume: int

class BusinessGoal(BaseModel):
    goal: str
    description: str

class ContentOutline(BaseModel):
    title: str
    sections: List[str]
    target_word_count: Optional[int] = None
    seo_keywords: Optional[List[SEOKeyword]] = None
    business_goals: Optional[List[BusinessGoal]] = None

class ContentSection(BaseModel):
    title: str
    content: str
    citations: List[str]

class BlogPost(BaseModel):
    title: str
    sections: List[ContentSection]
    meta_description: str
    target_keywords: List[str]
    estimated_read_time: int

class WriterState(TypedDict):
    content_query: str
    research_insights: List[str]
    business_goals: Optional[List[BusinessGoal]]
    target_read_time: Optional[int]
    seo_keywords: Optional[List[SEOKeyword]]
    content_outline: Optional[ContentOutline]
    blog_post: Optional[BlogPost]
    current_writing_step: str

