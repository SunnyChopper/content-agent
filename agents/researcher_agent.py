# Standard
from typing import List, Any, Optional
from pathlib import Path
import logging
import hashlib
import json
import sys
import os

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Langchain
from langchain_community.document_transformers import Html2TextTransformer
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_openai import ChatOpenAI
from langchain_core.tools import Tool

# Langgraph
from langgraph.graph import StateGraph, START, END

# Models
from models.states.workflows.blog_workflow_state import BlogWorkflowState
from models.states.agents.researcher_state import (
    ResearchInsights,
    ConfidenceLevel,
    FollowUpTopics,
    SearchQueries, 
    SearchResults,
    SearchResult,
    InsightPoint
)

from settings import MAX_RESEARCH_LOOPS, DEFAULT_OPENAI_CHEAP_MODEL

# Prompts
BASE_PROMPT_DIR: str = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "prompts", "researcher")
GENERATE_SEARCH_QUERIES_SYSTEM_PROMPT: str = os.path.join(BASE_PROMPT_DIR, "generate-queries", "system.txt")
GENERATE_SEARCH_QUERIES_HUMAN_PROMPT: str = os.path.join(BASE_PROMPT_DIR, "generate-queries", "human.txt")
ANALYZE_SEARCH_RESULTS_SYSTEM_PROMPT: str = os.path.join(BASE_PROMPT_DIR, "analyze-results", "system.txt")
ANALYZE_SEARCH_RESULTS_HUMAN_PROMPT: str = os.path.join(BASE_PROMPT_DIR, "analyze-results", "human.txt")
SELECT_SEARCH_RESULTS_SYSTEM_PROMPT: str = os.path.join(BASE_PROMPT_DIR, "select-results", "system.txt")
SELECT_SEARCH_RESULTS_HUMAN_PROMPT: str = os.path.join(BASE_PROMPT_DIR, "select-results", "human.txt")
DEDUPE_INSIGHTS_SYSTEM_PROMPT: str = os.path.join(BASE_PROMPT_DIR, "dedupe-insights", "system.txt")
DEDUPE_INSIGHTS_HUMAN_PROMPT: str = os.path.join(BASE_PROMPT_DIR, "dedupe-insights", "human.txt")
EVALUATE_CONFIDENCE_SYSTEM_PROMPT: str = os.path.join(BASE_PROMPT_DIR, "evaluate-confidence", "system.txt")
EVALUATE_CONFIDENCE_HUMAN_PROMPT: str = os.path.join(BASE_PROMPT_DIR, "evaluate-confidence", "human.txt")
GENERATE_FOLLOW_UP_TOPICS_SYSTEM_PROMPT: str = os.path.join(BASE_PROMPT_DIR, "generate-followups", "system.txt")
GENERATE_FOLLOW_UP_TOPICS_HUMAN_PROMPT: str = os.path.join(BASE_PROMPT_DIR, "generate-followups", "human.txt")

# Settings
CACHE_DIR: str = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache")
SEARCH_CACHE_DIR: str = os.path.join(CACHE_DIR, "search_results")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

class ResearchCache:
    @staticmethod
    def ensure_cache_dirs():
        """Ensure cache directories exist."""
        Path(CACHE_DIR).mkdir(exist_ok=True)
        Path(SEARCH_CACHE_DIR).mkdir(exist_ok=True)

    @staticmethod
    def get_cache_key(content_query: str) -> str:
        """Generate a cache key for content query."""
        return hashlib.sha256(content_query.encode()).hexdigest()
    
    @staticmethod
    def get_cache_path(content_query: str) -> str:
        """Get the path to the cache file."""
        return os.path.join(SEARCH_CACHE_DIR, f"{ResearchCache.get_cache_key(content_query)}.json")

    @staticmethod
    def load_search_results(content_query: str) -> Optional[List[Any]]:
        """Load cached search results for a content query."""
        cache_path = ResearchCache.get_cache_path(content_query)
        if os.path.exists(cache_path):
            with open(cache_path, "r") as file:
                return json.load(file)
        return None

    @staticmethod
    def save_search_results(content_query: str, results: List[Any]):
        """Save search results for a content query."""
        cache_path = ResearchCache.get_cache_path(content_query)
        existing_results = ResearchCache.load_search_results(content_query) or []
        existing_results.extend(results)  # Add new results to existing ones
        with open(cache_path, "w") as file:
            json.dump(existing_results, file)

class ResearcherAgent:
    def __init__(self, llm: ChatOpenAI, web_search: Tool):
        self.llm = llm
        self.web_search = web_search
        ResearchCache.ensure_cache_dirs()

    def __generate_research_queries(self, query: str, existing_queries: List[str], existing_insights: Optional[ResearchInsights]) -> SearchQueries:
        """Generates research queries based on the content query, existing queries, and existing insights."""
        if existing_insights is None:
            existing_insights = ResearchInsights(insights=[], hyperlinks=[], new_search_queries=[])
        logger.info(f"Generating research queries (with {len(existing_insights.insights)} existing insights)")
        
        # Read system and human prompts
        with open(GENERATE_SEARCH_QUERIES_SYSTEM_PROMPT, "r") as file:
            system_prompt = file.read()
        with open(GENERATE_SEARCH_QUERIES_HUMAN_PROMPT, "r") as file:
            human_prompt_template = file.read()
        
        # Format the human prompt
        formatted_insights = "\n".join(existing_insights.insights) if existing_insights.insights else "No existing insights."
        human_prompt = human_prompt_template.format(
            content_query=query,
            research_data=formatted_insights,
            search_queries=existing_queries
        )
        
        # Create structured output LLM
        structured_llm = self.llm.with_structured_output(SearchQueries)
        
        # Invoke with both system and human messages
        return structured_llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ])
    
    def __get_search_results(self, content_query: str, queries: List[str], ignore_cache: bool = False) -> List[SearchResult]:
        """Retrieves search results for a list of queries using the web search tool."""
        # 1. Check cache first using the content query
        if not ignore_cache:
            cached_results = ResearchCache.load_search_results(content_query)
            if cached_results is not None:
                logger.info(f"Using {len(cached_results)} cached search results.")
                return cached_results
        else:
            logger.info(f"Ignoring cache for content query.")
        
        # 2. Get search results if none found in cache
        search_results: List[SearchResult] = []
        for index, query in enumerate(queries):
            logger.info(f"Processing query ({index + 1}/{len(queries)}): {query}")
            raw_results: Any = self.web_search.invoke(query)
            logger.debug(f"Parsing string results into a list of dictionaries using the LLM")
            structured_llm = self.llm.with_structured_output(SearchResults)
            parsed_results = structured_llm.invoke([
                SystemMessage(content="Convert the search results string into a list of dictionaries with 'title', 'link', 'snippet', and an empty 'full_content' field."),
                HumanMessage(content=f"Search results:\n{raw_results}")
            ])
            results: List[SearchResult] = parsed_results.results
            logger.debug(f"Parsed results: {results}")
            search_results.extend(results)

        # 3. Save search results to cache
        ResearchCache.save_search_results(content_query, search_results)
        return search_results
    
    def __pull_full_content(self, selected_results: List[SearchResult]) -> List[SearchResult]:
        """Gets the full content from a list of URLs."""
        logger.info(f"Getting full content from {len(selected_results)} URLs")
        urls: List[str] = [result['link'] for result in selected_results]

        try:
            # Load HTML content asynchronously
            loader = AsyncHtmlLoader(urls)
            html_docs = loader.load()
            logger.debug(f"Loaded {len(html_docs)} HTML documents")
            
            # Transform HTML to readable text
            logger.debug(f"Transforming {len(html_docs)} HTML documents to readable text")
            html2text = Html2TextTransformer()
            docs = html2text.transform_documents(html_docs)
            logger.debug(f"Transformed {len(docs)} documents")
            
            # Combine original search snippets with full content
            enhanced_results = []
            for i, result in enumerate(selected_results):
                enhanced_results.append({
                    'title': result['title'],
                    'link': result['link'],
                    'snippet': result['snippet'],
                    'full_content': docs[i].page_content if i < len(docs) else ''
                })
                
            logger.info(f"Successfully loaded full content for {len(docs)} out of {len(selected_results)} URLs")
        except Exception as e:
            logger.warning(f"Error loading full content from URLs: {str(e)}. Falling back to snippets only.")
            enhanced_results = selected_results
        return enhanced_results
    
    def __select_relevant_search_results(self, content_query: str, existing_insights: ResearchInsights, search_results: List[SearchResult], max_urls: int = 15) -> List[SearchResult]:
        """Selects relevant search results based on the content query and existing insights."""
        logger.info(f"Selecting relevant search results")

        with open(SELECT_SEARCH_RESULTS_SYSTEM_PROMPT, "r") as file:
            system_prompt_template = file.read()
        with open(SELECT_SEARCH_RESULTS_HUMAN_PROMPT, "r") as file:
            human_prompt_template = file.read()

        formatted_results = "\n".join([
            f"Title: {result['title']}\nURL: {result['link']}\nSnippet: {result['snippet']}\nFull Content: {result['full_content'] or 'No full content available.'}"
            for result in search_results
        ])
        human_prompt = human_prompt_template.format(
            content_query=content_query,
            formatted_results=formatted_results,
            existing_insights=existing_insights,
            max_urls=max_urls
        )
        system_prompt = system_prompt_template.format(max_urls=max_urls)

        structured_llm = self.llm.with_structured_output(SearchResults)
        selected_results: SearchResults = structured_llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ])

        logger.info(f"Selected {len(selected_results.results)} relevant search results from {len(search_results)} search results. This is before pulling full content.")

        return self.__pull_full_content(selected_results.results)

    def __analyze_search_results(self, content_query: str, existing_insights: ResearchInsights, search_results: List[SearchResult]) -> ResearchInsights:
        """Analyzes search results to extract key insights."""
        logger.info(f"Analyzing {len(search_results)} search results")

        # Get relevant search results
        relevant_results: List[SearchResult] = self.__select_relevant_search_results(
            content_query=content_query,
            existing_insights=existing_insights,
            search_results=search_results
        )

        logger.info(f"Selected {len(relevant_results)} relevant search results")

        # Read and format prompts
        with open(ANALYZE_SEARCH_RESULTS_SYSTEM_PROMPT, "r") as file:
            system_prompt = file.read()
        with open(ANALYZE_SEARCH_RESULTS_HUMAN_PROMPT, "r") as file:
            human_prompt_template = file.read()

        # Analyze each result and extract insights
        results_analyzed: int = 0
        all_insights: ResearchInsights = ResearchInsights(insights=[], hyperlinks=[], new_search_queries=[])
        for index, result in enumerate(relevant_results):
            logger.info(f"Analyzing result ({index + 1}/{len(relevant_results)}): {result['title']}")
            logger.info(f"Snippet: {result['snippet']}")
            human_prompt = human_prompt_template.format(
                content_query=content_query, 
                title=result['title'], 
                link=result['link'], 
                snippet=result['snippet'], 
                full_content=result['full_content'][:80000]
            )

            # Create structured output LLM
            structured_llm = self.llm.with_structured_output(ResearchInsights)
            insights: ResearchInsights = structured_llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt)
            ])
            logger.info(f"Pulled {len(insights.insights)} insights from {result['title']}")
            all_insights += insights
            results_analyzed += 1

            # Check confidence level and break if we meet the threshold
            confidence_level: ConfidenceLevel = self.__evaluate_confidence_level(content_query, all_insights.insights)
            if confidence_level.confidence_score > 0.8:
                logger.info(f"Confidence level is {confidence_level.confidence_score * 100:.1f}%, stopping analysis of search results early.")
                break
            else:
                logger.info(f"Confidence level is {confidence_level.confidence_score * 100:.1f}%, continuing analysis.")

        logger.info(f"Pulled {len(all_insights.insights)} insights from {results_analyzed} search results")
        return all_insights

    def __deduplicate_insights(self, insights: ResearchInsights) -> ResearchInsights:
        """Deduplicates insights using semantic similarity."""
        logger.info(f"Deduplicating {len(insights.insights)} insights")

        # Read system and human prompts
        with open(DEDUPE_INSIGHTS_SYSTEM_PROMPT, "r") as file:
            system_prompt = file.read()
        with open(DEDUPE_INSIGHTS_HUMAN_PROMPT, "r") as file:
            human_prompt_template = file.read()

        # Format the human prompt
        human_prompt = human_prompt_template.format(insights=str(insights))

        structured_llm = self.llm.with_structured_output(ResearchInsights)
        return structured_llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ])

    def __evaluate_confidence_level(self, content_query: str, insights: List[InsightPoint]) -> ConfidenceLevel:
        """Evaluates the confidence level of the research insights."""
        logger.info(f"Evaluating confidence level for {len(insights)} insights")
        
        # Read system and human prompts
        with open(EVALUATE_CONFIDENCE_SYSTEM_PROMPT, "r") as file:
            system_prompt = file.read()
        with open(EVALUATE_CONFIDENCE_HUMAN_PROMPT, "r") as file:
            human_prompt_template = file.read()

        # Format the human prompt with hierarchical insights
        formatted_insights = "\n".join([
            f"Main Point: {insight.main_point}\n" + 
            "\n".join([f"• {point}" for point in insight.supporting_points]) + "\n"
            for insight in insights
        ])
        
        human_prompt = human_prompt_template.format(
            content_query=content_query,
            research_insights=formatted_insights
        )

        # Create structured output LLM
        structured_llm = self.llm.with_structured_output(ConfidenceLevel)
        confidence_eval = structured_llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ])

        # Log detailed evaluation feedback
        logger.debug("Confidence Evaluation:")
        for dimension, score in confidence_eval.dimension_scores.model_dump().items():
            logger.debug(f"{dimension.title().replace('_', ' ').capitalize()}: {score:.2f}")
            logger.debug(f"Feedback: {confidence_eval.dimension_feedback.model_dump()[dimension]}")
        logger.debug(f"Confidence Score: {confidence_eval.confidence_score:.2f}")
        
        return confidence_eval

    def __generate_follow_up_topics(self, content_query: str, insights: List[str]) -> List[str]:
        """Generates follow-up topics based on the content query and insights."""
        logger.info(f"Generating follow-up topics for {len(insights)} insights")

        with open(GENERATE_FOLLOW_UP_TOPICS_SYSTEM_PROMPT, "r") as file:
            system_prompt = file.read()
        with open(GENERATE_FOLLOW_UP_TOPICS_HUMAN_PROMPT, "r") as file:
            human_prompt_template = file.read()

        human_prompt = human_prompt_template.format(content_query=content_query, insights=str(insights))
        structured_llm = self.llm.with_structured_output(FollowUpTopics)
        return structured_llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ])

    def invoke(self, state: BlogWorkflowState) -> BlogWorkflowState:
        # 1. Extract current research progress
        existing_research_insights: Optional[ResearchInsights] = state.get("research_insights")
        if existing_research_insights is None:
            existing_research_insights = ResearchInsights(insights=[], hyperlinks=[], new_search_queries=[])
        content_query: str = state.get("content_query", "")
        existing_search_queries: List[str] = state.get("search_queries", [])
        new_search_queries: List[str] = state.get("new_search_queries", [])

        logger.debug(f"Content query: {content_query}")
        if existing_research_insights is not None:
            logger.debug(f"Current research insights: {existing_research_insights.insights if existing_research_insights.insights else 'None'}")

        # 2. Generate search queries based on current progress
        if not new_search_queries:
            generate_queries_result: SearchQueries = self.__generate_research_queries(
                query=content_query,
                existing_queries=existing_search_queries,
                existing_insights=existing_research_insights
            )
            logger.debug(f"Generated search queries: {generate_queries_result.generated_queries}")
            new_search_queries = generate_queries_result.generated_queries

        next_loop_search_queries: List[str] = []
        queries_used: List[str] = []
        for query in new_search_queries:
            # 3a. Get search results for the single query
            logger.info(f"Getting search results for query: {query}...")
            search_results: List[SearchResult] = self.__get_search_results(
                content_query=content_query,
                queries=[query],
                ignore_cache=True
            )
            queries_used.append(query)

            # 3b. Analyze search results to extract insights from the single query
            new_insights: ResearchInsights = self.__analyze_search_results(content_query, existing_research_insights, search_results)

            logger.info(f"{len(new_insights.insights)} insights pulled.")
            logger.debug(f"New insights from search results: {new_insights.insights}")

            logger.info(f"{len(new_insights.hyperlinks)} hyperlinks pulled.")
            logger.debug(f"New hyperlinks from search results: {new_insights.hyperlinks}")

            logger.info(f"{len(new_insights.new_search_queries)} new search queries pulled.")
            logger.debug(f"New search queries from search results: {new_insights.new_search_queries}")

            # 3c. Combine new insights with existing ones
            all_insights: ResearchInsights = existing_research_insights + new_insights
            next_loop_search_queries.extend(new_insights.new_search_queries)

            # 3d. Evaluate the confidence level of the new combined and deduplicated insights
            confidence_eval: ConfidenceLevel = self.__evaluate_confidence_level(content_query, all_insights.insights)

            # Stop if we have high confidence across all dimensions
            if (confidence_eval.confidence_score > 0.8 and 
                all(score > 0.7 for score in confidence_eval.dimension_scores.model_dump().values())):
                logger.info(f"Confidence level is {confidence_eval.confidence_score * 100:.1f}% with good coverage across all dimensions")
                break

        # 4. Deduplicate insights
        deduplicated_insights: ResearchInsights = self.__deduplicate_insights(all_insights)

        # 4. Generate follow-up topics if confidence is high, marking the end of the research loop
        follow_up_topics: List[str] = []
        if state.get("current_loop", 0) == (MAX_RESEARCH_LOOPS - 1):
            logger.info(f"Generating follow-up topics for {len(deduplicated_insights.insights)} insights")
            follow_up_topics_response = self.__generate_follow_up_topics(content_query, deduplicated_insights.insights)
            logger.info(f"Generated {len(follow_up_topics_response.topics)} follow-up topics: {follow_up_topics_response.topics}")
            follow_up_topics = follow_up_topics_response.topics

        # 5. Update the state with new insights and hyperlinks
        return BlogWorkflowState(
            messages=state.get("messages", []),
            research_insights=deduplicated_insights,
            hyperlinks=deduplicated_insights.hyperlinks,
            search_queries=existing_search_queries + queries_used,
            new_search_queries=next_loop_search_queries,
            confidence_level=confidence_eval.confidence_score,
            content_query=content_query,
            current_loop=state.get("current_loop", 0) + 1,
            follow_up_topics=follow_up_topics,
            business_goals=state.get("business_goals", []),
            target_read_time=state.get("target_read_time", None),
            seo_keywords=state.get("seo_keywords", []),
            content_outline=state.get("content_outline", None),
            blog_post=state.get("blog_post", None),
            current_writing_step=state.get("current_writing_step", ""),
            editor_insights=state.get("editor_insights", None),
            editor_feedback=state.get("editor_feedback", None),
            editor_suggestions=state.get("editor_suggestions", None),
            editor_suggestions_applied=state.get("editor_suggestions_applied", None)
        )
    
def researcher_agent_postaction(state: BlogWorkflowState) -> str:
    has_confidence: bool = state.get("confidence_level", 0) > 0.8
    current_loop: int = state.get("current_loop", 0)
    has_new_search_queries: bool = len(state.get("new_search_queries", [])) > 0

    if (current_loop >= MAX_RESEARCH_LOOPS or not has_new_search_queries) and has_confidence:
        return END
    elif has_new_search_queries or not has_confidence:
        return "researcher"
    else:
        return END

workflow = StateGraph(BlogWorkflowState)
search_api = DuckDuckGoSearchAPIWrapper(max_results=5)
researcher_agent = ResearcherAgent(
    llm=ChatOpenAI(model=DEFAULT_OPENAI_CHEAP_MODEL, temperature=0),
    web_search=DuckDuckGoSearchResults(api_wrapper=search_api)
)

# Nodes
workflow.add_node("researcher", researcher_agent.invoke)

# Edges
workflow.add_edge(START, "researcher")
workflow.add_conditional_edges("researcher", researcher_agent_postaction)

researcher_workflow = workflow.compile()

if __name__ == "__main__":
    def print_result(result: BlogWorkflowState):
        content_query: str = result.get("content_query", "🚨 Could not retrieve content query.")
        search_queries: List[str] = result.get("search_queries", [])

        print("\n=== Research Results ===\n")
        print("📝 Content Query:")
        print(f"{content_query}\n")
        
        print("🔍 Search Queries Used:")
        for i, query in enumerate(search_queries, 1):
            print(f"{i}. {query}")
        print()
        
        print("💡 Research Insights:")
        for i, insight in enumerate(result['research_insights'].insights, 1):
            print(f"{i}. {insight.main_point}")
            for point in insight.supporting_points:
                print(f"   • {point}")
            print()
        
        print("🔗 Supporting References:")
        for i, link in enumerate(result['hyperlinks'], 1):
            print(f"{i}. {link['title']}")
            print(f"   URL: {link['link']}")
            print(f"   Summary: {link['snippet']}\n")
        
        print("🎯 Confidence Level:")
        print(f"{result['confidence_level'] * 100:.1f}%")
        
        if result.get('follow_up_topics'):
            print("\n📌 Follow-Up Topics:")
            for i, topic in enumerate(result['follow_up_topics'], 1):
                print(f"{i}. {topic}")
        print()

    logger.info("Starting Researcher Agent")
    result: BlogWorkflowState = researcher_workflow.invoke({"content_query": "What are 10 realistic vertical AI agent projects that could be profitable for solo founders in 2025?"})
    print_result(result)
