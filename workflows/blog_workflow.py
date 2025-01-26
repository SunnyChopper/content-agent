from typing import Dict, Any, Union
import logging
import sys
import os
import json
from datetime import datetime

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaLLM

from agents.researcher_agent import ResearcherAgent
from agents.writer_agent import WriterAgent
from agents.editor_agent import EditorAgent
from models.states.agents.writer_state import WriterState, BusinessGoal, SEOKeyword, BlogPost
from models.states.workflows.blog_workflow_state import BlogWorkflowState
from models.states.agents.researcher_state import ResearcherState

from settings import DEFAULT_OPENAI_CHEAP_MODEL, MAX_RESEARCH_LOOPS, DEFAULT_OLLAMA_MODEL

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
    
    # Convert structured insights to list of strings
    research_insights = []
    if state.get("research_insights"):
        for insight in state["research_insights"].insights:
            research_insights.append(f"Main point: {insight.main_point}")
            research_insights.extend([f"  - Supporting point: {point}" for point in insight.supporting_points])
    
    return {
        "content_query": state["content_query"],
        "research_insights": research_insights or ["No research insights found"],
        "business_goals": state.get("business_goals", []),
        "target_read_time": state.get("target_read_time"),
        "seo_keywords": state.get("seo_keywords", []),
        "current_writing_step": "generate_outline",
        # Carry forward existing writing state if present
        "content_outline": state.get("content_outline"),
        "blog_post": state.get("blog_post")
    }

def create_blog_workflow(
    research_llm: Union[ChatOpenAI, OllamaLLM],
    writing_llm: Union[ChatOpenAI, OllamaLLM],
    skip_research: bool = False
) -> StateGraph:
    """Creates a workflow that combines research and writing."""
    workflow = StateGraph(BlogWorkflowState)
    
    # Create agents
    search_api = DuckDuckGoSearchAPIWrapper(max_results=5)
    researcher_agent = ResearcherAgent(llm=research_llm, web_search=DuckDuckGoSearchResults(api_wrapper=search_api))
    writer_agent = WriterAgent(llm=writing_llm)
    editor_agent = EditorAgent(llm=writing_llm)
    
    # Add nodes
    workflow.add_node("researcher", researcher_agent.invoke)
    workflow.add_node("writer", writer_agent.invoke)
    workflow.add_node("editor", editor_agent.invoke)
    
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
            logger.info("Writing complete, moving to editor")
            logger.info(f"Post draft: {state['blog_post']}")
            return "editor"
        return "writer"
    workflow.add_conditional_edges("writer", writer_router)
    
    def editor_router(state: Dict[str, Any]) -> str:
        # Check if editor suggestions have been applied
        if state.get("editor_suggestions_applied"):
            return END
        return "editor"
    workflow.add_conditional_edges("editor", editor_router)
    
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
        "blog_post": BlogPost(
            title="",
            sections=[],
            meta_description="",
            target_keywords=[],
            estimated_read_time=0
        ),
        "current_writing_step": "generate_outline",
        "editor_insights": None,
        "editor_feedback": None,
        "editor_suggestions": None,
        "editor_suggestions_applied": None
    }

def save_blog_post(state: BlogWorkflowState, filename: str = "generated_blog_post.txt"):
    """Enhanced saving with section validation"""
    with open(filename, "w", encoding="utf-8") as f:
        # Write formatted blog post
        f.write("=== FORMATTED BLOG POST ===\n\n")
        
        if state.get('blog_post'):
            blog = state['blog_post']
            f.write(f"Title: {blog.title}\n\n")
            f.write(f"Meta Description: {blog.meta_description}\n\n")
            f.write(f"Estimated Read Time: {blog.estimated_read_time} minutes\n\n")
            
            for section in blog.sections:
                f.write(f"## {section.title}\n")
                f.write(f"{section.content}\n\n")
                if section.citations:
                    f.write("Citations:\n")
                    for citation in section.citations:
                        f.write(f"- {citation}\n")
                f.write("\n")
        
        # Write raw state data
        f.write("\n\n=== RAW STATE DATA ===\n")
        f.write(json.dumps({
            k: v.model_dump() if hasattr(v, "model_dump") else v
            for k, v in state.items()
        }, indent=2, default=str))

        # Add section validation
        if blog:
            if not blog.sections:
                f.write("⚠️ No sections generated - possible workflow error\n")
            elif not any(section.content.strip() for section in blog.sections):
                f.write("⚠️ Sections exist but contain no content - check writer agent\n")

if __name__ == "__main__":
    # Create workflow
    local_writing_llm = ChatOllama(model=DEFAULT_OLLAMA_MODEL, temperature=0)
    local_research_llm = ChatOllama(model=DEFAULT_OLLAMA_MODEL, temperature=0)
    writing_llm = ChatOpenAI(base_url="https://api.deepseek.com/v1", api_key="sk-6fbfef6c01494c2b9a97d65cf7f8e33c", model="deepseek-chat", temperature=0)
    research_llm = ChatOpenAI(base_url="https://api.deepseek.com/v1", api_key="sk-6fbfef6c01494c2b9a97d65cf7f8e33c", model="deepseek-chat", temperature=0)
    # research_llm = ChatOpenAI(model=DEFAULT_OPENAI_CHEAP_MODEL, temperature=0)
    workflow = create_blog_workflow(local_research_llm, local_writing_llm, skip_research=False)
    
    # Example input state
    input_state = create_initial_blog_workflow_state()
    input_state["content_query"] = "Advanced LangGraph patterns for building robust AI agent workflows"
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
    final_state = input_state
    for step in workflow.stream(input_state):
        if "blog_post" in step:
            print(f"New section: {step['blog_post'].sections[-1].title}")
            final_state = step  # Capture the latest state
        if "content_outline" in step: 
            print("Outline generated!")
            final_state = step  # Capture the latest state

    # Print results from FINAL STATE
    print("\n=== Blog Generation Results ===\n")
    
    print("📝 Content Query:")
    print(f"{final_state['content_query']}\n")
    
    if final_state.get('research_insights'):
        print("🔍 Research Insights:")
        for i, insight in enumerate(final_state['research_insights'].insights, 1):
            print(f"{i}. {insight.main_point}")
            for point in insight.supporting_points:
                print(f"   • {point}")
            print()
    
    if final_state.get('content_outline'):
        print("📋 Content Outline:")
        print(f"Title: {final_state['content_outline'].title}")
        for i, section in enumerate(final_state['content_outline'].sections, 1):
            print(f"{i}. {section}")
        print()
    
    if final_state.get('blog_post'):
        print("📚 Blog Post:")
        print(f"Title: {final_state['blog_post'].title}")
        print(f"Meta Description: {final_state['blog_post'].meta_description}")
        print(f"Estimated Read Time: {final_state['blog_post'].estimated_read_time} minutes\n")
        
        for section in final_state['blog_post'].sections:
            print(f"## {section.title}")
            print(f"{section.content}\n")
            if section.citations:
                print("Citations:")
                for citation in section.citations:
                    print(f"- {citation}")
            print()
    
    if final_state.get('seo_keywords'):
        print("🎯 Target Keywords:")
        for kw in final_state['seo_keywords']:
            print(f"- {kw.keyword} ({kw.search_volume} monthly searches)")
        print()

    # Save results to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"blog_post_{timestamp}.txt"
    save_blog_post(final_state, filename)
    print(f"\n💾 Blog post saved to {filename}") 