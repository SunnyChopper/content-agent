from typing import Dict, Any
import logging
import sys
import os

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.tools import DuckDuckGoSearchResults
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI

from agents.researcher_agent import ResearcherAgent
from agents.writer_agent import WriterAgent
from models.states.agents.writer_state import WriterState, BusinessGoal, SEOKeyword
from models.states.workflows.blog_workflow_state import BlogWorkflowState
from models.states.agents.researcher_state import ResearcherState

from settings import DEFAULT_OPENAI_CHEAP_MODEL, MAX_RESEARCH_LOOPS

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def research_to_writing_state_converter(state: Dict[str, Any]) -> Dict[str, Any]:
    """Converts research state to writing state."""
    logger.info("Converting research state to writing state")
    
    research_insights = []
    for insight in state["research_insights"].insights:
        research_insights.append(insight.main_point)
        research_insights.extend([f"  - {point}" for point in insight.supporting_points])
    
    return {
        "content_query": state["content_query"],
        "research_insights": research_insights,
        "business_goals": state.get("business_goals"),
        "target_read_time": state.get("target_read_time"),
        "seo_keywords": state.get("seo_keywords"),
        "current_writing_step": "generate_outline"
    }

def create_blog_workflow(
    llm: ChatOpenAI = ChatOpenAI(model=DEFAULT_OPENAI_CHEAP_MODEL, temperature=0),
    skip_research: bool = False
) -> StateGraph:
    """Creates a workflow that combines research and writing."""
    workflow = StateGraph(BlogWorkflowState)
    
    # Create agents
    search_api = DuckDuckGoSearchAPIWrapper(max_results=5)
    researcher_agent = ResearcherAgent(llm=llm, web_search=DuckDuckGoSearchResults(api_wrapper=search_api))
    writer_agent = WriterAgent(llm)
    
    # Add nodes
    workflow.add_node("researcher", researcher_agent.invoke)
    workflow.add_node("writer", writer_agent.invoke)
    
    # Add edges
    if not skip_research:
        workflow.add_edge(START, "researcher")
    else:
        workflow.add_edge(START, "writer")
    
    def researcher_router(state: ResearcherState) -> str:
        if state.get("current_loop", 0) >= MAX_RESEARCH_LOOPS or state.get("confidence_level", 0) > 0.8:
            return "writer"
        return "researcher"
    workflow.add_conditional_edges("researcher", researcher_router)
    
    def writer_router(state: Dict[str, Any]) -> str:
        if state.get("current_writing_step") == "complete":
            return END
        return "writer"
    workflow.add_conditional_edges("writer", writer_router)
    
    return workflow.compile()

def create_initial_blog_workflow_state() -> BlogWorkflowState:
    return {
        "messages": [],
        "research_insights": None,
        "search_queries": [],
        "new_search_queries": [],
        "confidence_level": 0.0,
        "hyperlinks": [],
        "content_query": "",
        "current_loop": 0,
        "follow_up_topics": None,
        "business_goals": [],
        "target_read_time": None,
        "seo_keywords": [],
        "content_outline": None,
        "blog_post": None,
        "current_writing_step": "generate_outline",
        "editor_insights": None,
        "editor_feedback": None,
        "editor_suggestions": None,
        "editor_suggestions_applied": None
    }

if __name__ == "__main__":
    # Create workflow
    workflow = create_blog_workflow(skip_research=False)
    
    # Example input state
    input_state = create_initial_blog_workflow_state()
    input_state["content_query"] = "What are 10 realistic vertical AI agent projects that could be profitable for solo founders in 2025?"
    input_state["business_goals"] = [
        BusinessGoal(goal="Newsletter Signups", description="Leverage the article to generate newsletter signups for 'Amplify with AI', my online brand for everything related to building with AI.")
    ]
    input_state["target_read_time"] = 8
    input_state["seo_keywords"] = [
        SEOKeyword(keyword="ai agents", search_volume=50000),
        SEOKeyword(keyword="intelligent agent in ai", search_volume=50000),
        SEOKeyword(keyword="ai intelligent agent", search_volume=50000),
        SEOKeyword(keyword="ai industry trends", search_volume=5000)
    ]
    
    # Run workflow
    result = workflow.invoke(input_state)
    
    # Print results
    print("\n=== Blog Generation Results ===\n")
    
    print("📝 Content Query:")
    print(f"{result['content_query']}\n")
    
    if result.get('research_insights'):
        print("🔍 Research Insights:")
        for i, insight in enumerate(result['research_insights'], 1):
            print(f"{i}. {insight}")
        print()
    
    if result.get('content_outline'):
        print("📋 Content Outline:")
        print(f"Title: {result['content_outline'].title}")
        for i, section in enumerate(result['content_outline'].sections, 1):
            print(f"{i}. {section}")
        print()
    
    if result.get('blog_post'):
        print("📚 Blog Post:")
        print(f"Title: {result['blog_post'].title}")
        print(f"Meta Description: {result['blog_post'].meta_description}")
        print(f"Estimated Read Time: {result['blog_post'].estimated_read_time} minutes\n")
        
        for section in result['blog_post'].sections:
            print(f"## {section.title}")
            print(f"{section.content}\n")
            if section.citations:
                print("Citations:")
                for citation in section.citations:
                    print(f"- {citation}")
            print()
    
    if result.get('seo_keywords'):
        print("🎯 Target Keywords:")
        for kw in result['seo_keywords']:
            print(f"- {kw.keyword} ({kw.search_volume} monthly searches)")
        print() 