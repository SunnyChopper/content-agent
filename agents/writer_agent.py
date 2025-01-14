import os
import sys
import logging
from typing import List, Optional

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

from models.states.agents.writer_state import (
    WriterState,
    ContentOutline,
    BlogPost,
    BusinessGoal,
    SEOKeyword,
    ContentSection
)

from settings import DEFAULT_OPENAI_CHEAP_MODEL

# Prompts
BASE_PROMPT_DIR: str = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "prompts", "writer")
GENERATE_OUTLINE_SYSTEM_PROMPT: str = os.path.join(BASE_PROMPT_DIR, "generate-outline", "system.txt")
GENERATE_OUTLINE_HUMAN_PROMPT: str = os.path.join(BASE_PROMPT_DIR, "generate-outline", "human.txt")
WRITE_SECTION_SYSTEM_PROMPT: str = os.path.join(BASE_PROMPT_DIR, "write-section", "system.txt")
WRITE_SECTION_HUMAN_PROMPT: str = os.path.join(BASE_PROMPT_DIR, "write-section", "human.txt")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

class WriterAgent:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

    def __generate_outline(
        self,
        content_query: str,
        research_insights: List[str],
        business_goals: Optional[List[BusinessGoal]] = None,
        target_read_time: Optional[int] = None,
        seo_keywords: Optional[List[SEOKeyword]] = None
    ) -> ContentOutline:
        """Generates a content outline based on research insights and requirements."""
        logger.info("Generating content outline")
        
        # Read system and human prompts
        with open(GENERATE_OUTLINE_SYSTEM_PROMPT, "r") as file:
            system_prompt = file.read()
        with open(GENERATE_OUTLINE_HUMAN_PROMPT, "r") as file:
            human_prompt_template = file.read()
        
        # Format insights and requirements
        formatted_insights = "\n".join(f"- {insight}" for insight in research_insights)
        formatted_goals = "\n".join(f"- {goal.goal}: {goal.description}" for goal in (business_goals or []))
        formatted_keywords = "\n".join(f"- {kw.keyword} ({kw.search_volume} monthly searches)" for kw in (seo_keywords or []))
        
        # Calculate target word count if read time is specified (assuming 200 words per minute)
        target_word_count = target_read_time * 200 if target_read_time else None
        
        human_prompt = human_prompt_template.format(
            content_query=content_query,
            research_insights=formatted_insights,
            business_goals=formatted_goals or "No specific business goals provided.",
            seo_keywords=formatted_keywords or "No SEO keywords provided.",
            target_word_count=f"Target word count: {target_word_count} words" if target_word_count else "No specific word count target."
        )
        
        # Create structured output LLM
        structured_llm = self.llm.with_structured_output(ContentOutline)
        
        # Generate outline
        return structured_llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ])

    def __write_section(
        self,
        section_title: str,
        content_query: str,
        research_insights: List[str],
        outline: ContentOutline,
        business_goals: Optional[List[BusinessGoal]] = None,
        seo_keywords: Optional[List[SEOKeyword]] = None
    ) -> ContentSection:
        """Writes a single section of the blog post."""
        logger.info(f"Writing section: {section_title}")
        
        # Read system and human prompts
        with open(WRITE_SECTION_SYSTEM_PROMPT, "r") as file:
            system_prompt = file.read()
        with open(WRITE_SECTION_HUMAN_PROMPT, "r") as file:
            human_prompt_template = file.read()
        
        # Format insights and requirements
        formatted_insights = "\n".join(f"- {insight}" for insight in research_insights)
        formatted_goals = "\n".join(f"- {goal.goal}: {goal.description}" for goal in (business_goals or []))
        formatted_keywords = "\n".join(f"- {kw.keyword} ({kw.search_volume} monthly searches)" for kw in (seo_keywords or []))
        formatted_outline = "\n".join(f"- {section}" for section in outline.sections)
        
        human_prompt = human_prompt_template.format(
            content_query=content_query,
            section_title=section_title,
            research_insights=formatted_insights,
            business_goals=formatted_goals or "No specific business goals provided.",
            seo_keywords=formatted_keywords or "No SEO keywords provided.",
            content_outline=formatted_outline
        )
        
        # Create structured output LLM
        structured_llm = self.llm.with_structured_output(ContentSection)
        
        # Write section
        return structured_llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ])

    def __generate_meta_description(self, content_query: str, outline: ContentOutline, seo_keywords: Optional[List[SEOKeyword]] = None) -> str:
        """Generates an SEO-optimized meta description."""
        # Get primary keyword (highest search volume)
        primary_keyword = None
        if seo_keywords:
            primary_keyword = max(seo_keywords, key=lambda k: k.search_volume).keyword
        
        # Create compelling meta description with primary keyword front-loaded if available
        if primary_keyword:
            # Ensure the primary keyword is at the start of the description
            if not content_query.lower().startswith(primary_keyword.lower()):
                meta_description = f"Discover {primary_keyword}: {content_query}"
            else:
                meta_description = f"Discover {content_query}"
        else:
            meta_description = f"Learn about {content_query}"
            
        # Add value proposition
        meta_description += " in this comprehensive guide. "
        
        # Add preview of what they'll learn (from first few sections)
        preview_sections = outline.sections[:2]  # Take first two sections
        if preview_sections:
            meta_description += f"Find out {preview_sections[0].lower()}"
            if len(preview_sections) > 1:
                meta_description += f" and {preview_sections[1].lower()}"
            meta_description += "."
            
        # Ensure it's not too long (optimal length is 150-160 characters)
        if len(meta_description) > 155:
            meta_description = meta_description[:152] + "..."
            
        return meta_description

    def invoke(self, state: WriterState) -> WriterState:
        current_writing_step = state.get("current_writing_step", "generate_outline")
        content_query = state["content_query"]
        research_insights = state["research_insights"]
        business_goals = state.get("business_goals")
        target_read_time = state.get("target_read_time")
        seo_keywords = state.get("seo_keywords")
        
        if current_writing_step == "generate_outline":
            # Generate content outline
            outline = self.__generate_outline(
                content_query=content_query,
                research_insights=research_insights,
                business_goals=business_goals,
                target_read_time=target_read_time,
                seo_keywords=seo_keywords
            )
            
            return WriterState(
                content_query=content_query,
                research_insights=research_insights,
                business_goals=business_goals,
                target_read_time=target_read_time,
                seo_keywords=seo_keywords,
                content_outline=outline,
                current_writing_step="write_sections"
            )
            
        elif current_writing_step == "write_sections":
            outline = state["content_outline"]
            sections = []
            
            # Write each section
            for section_title in outline.sections:
                section = self.__write_section(
                    section_title=section_title,
                    content_query=content_query,
                    research_insights=research_insights,
                    outline=outline,
                    business_goals=business_goals,
                    seo_keywords=seo_keywords
                )
                sections.append(section)
            
            # Calculate estimated read time (200 words per minute)
            total_words = sum(len(section.content.split()) for section in sections)
            estimated_read_time = total_words // 200
            
            # Generate SEO-optimized meta description
            meta_description = self.__generate_meta_description(
                content_query=content_query,
                outline=outline,
                seo_keywords=seo_keywords
            )
            
            # Create final blog post
            blog_post = BlogPost(
                title=outline.title,
                sections=sections,
                meta_description=meta_description,
                target_keywords=[kw.keyword for kw in (seo_keywords or [])],
                estimated_read_time=estimated_read_time
            )
            
            return WriterState(
                content_query=content_query,
                research_insights=research_insights,
                business_goals=business_goals,
                target_read_time=target_read_time,
                seo_keywords=seo_keywords,
                content_outline=outline,
                blog_post=blog_post,
                current_writing_step="complete"
            )
        
        return state

def writer_agent_postaction(state: WriterState) -> str:
    current_writing_step = state.get("current_writing_step", "generate_outline")
    
    if current_writing_step == "complete":
        return END
    elif current_writing_step == "generate_outline":
        return "writer"
    elif current_writing_step == "write_sections":
        return "writer"
    else:
        return END

workflow = StateGraph(WriterState)
writer_agent = WriterAgent(
    llm=ChatOpenAI(model=DEFAULT_OPENAI_CHEAP_MODEL, temperature=0.7)
)

# Nodes
workflow.add_node("writer", writer_agent.invoke)

# Edges
workflow.add_edge(START, "writer")
workflow.add_conditional_edges("writer", writer_agent_postaction)

writer_workflow = workflow.compile()

if __name__ == "__main__":
    def print_result(result: WriterState):
        print("\n=== Content Generation Results ===\n")
        
        print("📝 Content Query:")
        print(f"{result['content_query']}\n")
        
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
    
    # Example usage
    test_state = WriterState(
        content_query="What are 10 realistic vertical AI agent projects that could be profitable for solo founders in 2025?",
        research_insights=[
            {
                "main_point": "Vertical AI Agents are specialized systems designed to automate specific tasks within industries",
                "supporting_points": [
                    "They focus on solving specific industry problems rather than being general-purpose",
                    "They can achieve higher performance by specializing in a narrow domain",
                    "They require less training data compared to general AI systems",
                    "They can be developed and deployed more quickly due to their focused scope"
                ]
            },
            {
                "main_point": "The market for Vertical AI is expected to grow dramatically by 2025",
                "supporting_points": [
                    "Industry analysts predict significant market expansion in specialized AI solutions",
                    "Businesses are increasingly seeking AI solutions for specific operational challenges",
                    "The trend is moving away from general AI towards specialized solutions",
                    "Early movers in vertical AI markets can establish strong competitive positions"
                ]
            },
            {
                "main_point": "Solo founders have unique advantages in the Vertical AI space",
                "supporting_points": [
                    "Lower capital requirements compared to general AI development",
                    "Ability to move quickly and adapt to market needs",
                    "Can leverage existing open-source models and tools",
                    "Can focus on underserved niches in specific industries"
                ]
            },
            {
                "main_point": "Vertical AI Agents offer compelling ROI for businesses",
                "supporting_points": [
                    "Significant reduction in operational costs",
                    "Improved accuracy and consistency in task execution",
                    "24/7 availability without human limitations",
                    "Scalability without proportional cost increases"
                ]
            },
            {
                "main_point": "Integration capabilities are crucial for success",
                "supporting_points": [
                    "Must work seamlessly with existing business systems",
                    "APIs and webhooks are essential for connectivity",
                    "Should support standard data formats and protocols",
                    "Need to provide real-time data synchronization"
                ]
            },
            {
                "main_point": "Sustainability and resource optimization are key opportunities",
                "supporting_points": [
                    "Growing demand for AI in environmental monitoring",
                    "Need for smart resource management systems",
                    "Increasing focus on carbon footprint reduction",
                    "Regulatory compliance driving adoption"
                ]
            },
            {
                "main_point": "User experience is becoming more sophisticated",
                "supporting_points": [
                    "Trend towards more natural language interactions",
                    "Increasing demand for personalized experiences",
                    "Need for context-aware responses",
                    "Focus on reducing friction in user interactions"
                ]
            },
            {
                "main_point": "Open-source AI is changing the development landscape",
                "supporting_points": [
                    "More powerful models becoming freely available",
                    "Reduced barriers to entry for AI development",
                    "Growing ecosystem of tools and libraries",
                    "Community-driven improvements and innovations"
                ]
            },
            {
                "main_point": "Cybersecurity presents both challenges and opportunities",
                "supporting_points": [
                    "Need for AI-powered security solutions",
                    "Growing concerns about AI-based attacks",
                    "Opportunity for specialized security agents",
                    "Focus on privacy-preserving AI solutions"
                ]
            },
            {
                "main_point": "Real-time processing is becoming essential",
                "supporting_points": [
                    "Businesses need immediate insights and responses",
                    "Edge computing enabling faster processing",
                    "Growing demand for real-time decision support",
                    "Need for continuous monitoring and adaptation"
                ]
            }
        ],
        business_goals=[
            BusinessGoal(goal="Newsletter Signups", description="Leverage the article to generate newsletter signups for 'Amplify with AI', my online brand for everything related to building with AI.")
        ],
        target_read_time=8,
        seo_keywords=[
            SEOKeyword(keyword="ai agents", search_volume=50000),
            SEOKeyword(keyword="intelligent agent in ai", search_volume=50000),
            SEOKeyword(keyword="ai intelligent agent", search_volume=50000),
            SEOKeyword(keyword="ai industry trends", search_volume=5000)
        ],
        current_writing_step="generate_outline"
    )
    
    result = writer_workflow.invoke(test_state)
    print_result(result) 