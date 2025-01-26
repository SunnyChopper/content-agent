from typing import Optional, TypedDict, Annotated, Literal
import logging
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from models.states.agents.writer_state import BlogPost
from models.states.workflows.blog_workflow_state import BlogWorkflowState
import copy
from pydantic import BaseModel, validator
from pathlib import Path
import os
from langgraph.prebuilt import ValidationNode

logger = logging.getLogger(__name__)

# Move validation classes outside the class
class SimilarityCheck(BaseModel):
    """Validate similarity between two pieces of text."""
    is_similar: bool
    
    @validator("is_similar")
    def validate_similarity(cls, v):
        if not isinstance(v, bool):
            raise ValueError("Similarity check must return a boolean")
        return v

class CitationValidation(BaseModel):
    """Validate citation format and content."""
    is_valid: bool
    formatted_citation: str

class CoherenceValidation(BaseModel):
    """Validate content coherence."""
    is_coherent: bool
    feedback: str

class EditorAgent:
    # Get the project root directory (2 levels up from this file)
    PROJECT_ROOT = Path(__file__).parent.parent
    
    # Define prompt paths relative to project root
    REVISE_CONTENT_SYSTEM_PROMPT = PROJECT_ROOT / "prompts" / "editor" / "revise-content" / "system.txt"
    REVISE_CONTENT_HUMAN_PROMPT = PROJECT_ROOT / "prompts" / "editor" / "revise-content" / "human.txt"

    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.max_content_length = 15000  # Adjust based on your LLM's token limit
        
        # Validate prompt files exist
        if not self.REVISE_CONTENT_SYSTEM_PROMPT.exists():
            raise FileNotFoundError(f"System prompt file not found at {self.REVISE_CONTENT_SYSTEM_PROMPT}")
        if not self.REVISE_CONTENT_HUMAN_PROMPT.exists():
            raise FileNotFoundError(f"Human prompt file not found at {self.REVISE_CONTENT_HUMAN_PROMPT}")

    def _split_content(self, blog_post: BlogPost) -> list[BlogPost]:
        """Preserve existing content when splitting"""
        chunks = []
        current_chunk = copy.deepcopy(blog_post)  # Carry forward all metadata
        current_chunk.sections = []
        
        current_length = 0
        
        for section in blog_post.sections:
            if current_length + len(section.content) > self.max_content_length:
                chunks.append(current_chunk)
                current_chunk = copy.deepcopy(blog_post)  # Maintain metadata
                current_chunk.sections = []
                current_length = 0
            
            current_chunk.sections.append(section)
            current_length += len(section.content)
        
        if current_chunk.sections:
            chunks.append(current_chunk)
            
        return chunks

    def revise_content(self, blog_post: BlogPost, target_read_time: Optional[int] = None) -> BlogPost:
        """Add content validation before revision"""
        if not blog_post.sections:
            logger.error("No sections to revise!")
            return blog_post
        
        if sum(len(section.content) for section in blog_post.sections) == 0:
            logger.error("All sections empty!")
            return blog_post

        # Proceed with revision
        logger.info("Revising blog post content")

        # Initialize global tracking of seen titles and contents
        self._seen_titles = []
        self._seen_contents = []

        # Split content if too large
        if sum(len(section.content) for section in blog_post.sections) > self.max_content_length:
            chunks = self._split_content(blog_post)
            revised_sections = []
            
            for chunk in chunks:
                revised_chunk = self._revise_chunk(chunk, target_read_time)
                # Only extend with non-duplicate sections
                for section in revised_chunk.sections:
                    is_duplicate = False
                    for seen_title, seen_content in zip(self._seen_titles, self._seen_contents):
                        if self._is_similar_title(section.title, seen_title) or self._is_similar_content(section.content, seen_content):
                            logger.warning(f"Detected cross-chunk duplicate section: '{section.title}' is similar to '{seen_title}'")
                            is_duplicate = True
                            break
                    
                    if not is_duplicate:
                        revised_sections.append(section)
                        self._seen_titles.append(section.title)
                        self._seen_contents.append(section.content)
            
            # Clean up global state
            del self._seen_titles
            del self._seen_contents
            
            return BlogPost(
                title=blog_post.title,
                sections=revised_sections,
                meta_description=blog_post.meta_description,
                target_keywords=blog_post.target_keywords,
                estimated_read_time=blog_post.estimated_read_time
            )
        else:
            result = self._revise_chunk(blog_post, target_read_time)
            # Clean up global state
            del self._seen_titles
            del self._seen_contents
            return result

    def _is_similar_title(self, title1: str, title2: str) -> bool:
        """Check if two titles are semantically similar using LLM."""
        system_prompt = """You are a semantic similarity detector. Your task is to determine if two titles are semantically similar enough that they would create redundancy in a blog post.
        Consider:
        1. Core topic overlap
        2. Main concepts being discussed
        3. Overall intent of the section
        Do not consider minor variations in wording if the core meaning is the same.
        
        Return a JSON object with a single boolean field "is_similar"."""

        human_prompt = f"""Title 1: {title1}
Title 2: {title2}

Are these titles semantically similar enough that having both would create redundancy in a blog post?"""

        try:
            structured_llm = self.llm.with_structured_output(SimilarityCheck)
            result = structured_llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt)
            ])
            return result.is_similar
        except Exception as e:
            logger.warning(f"Failed to check title similarity via LLM, falling back to basic check: {str(e)}")
            # Fallback to basic exact match if LLM fails
            return title1.lower().strip() == title2.lower().strip()

    def _validate_citation(self, citation: str) -> tuple[bool, str]:
        """Validate and clean up citation format using LLM."""
        system_prompt = """You are a citation validator for blog posts. Your task is to:
        1. Verify the citation follows the format: [Source: Document Title](url)
        2. Ensure the URL is valid (starts with http/https)
        3. Check that the document title is descriptive and relevant
        
        Return a JSON object with:
        - is_valid: boolean indicating if citation is valid or fixable
        - formatted_citation: the properly formatted citation"""

        human_prompt = f"""Citation to validate: {citation}"""

        try:
            structured_llm = self.llm.with_structured_output(CitationValidation)
            result = structured_llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt)
            ])
            return result.is_valid, result.formatted_citation
        except Exception as e:
            logger.warning(f"Failed to validate citation via LLM: {str(e)}")
            # Fallback to basic validation
            if citation.startswith("[Source:") and citation.endswith(")") and "http" in citation:
                return True, citation
            return False, citation

    def _validate_coherence(self, current_section: str, previous_section: Optional[str] = None) -> tuple[bool, str]:
        """Validate coherence and transitions between sections using LLM."""
        system_prompt = """You are a content coherence validator. Your task is to:
        1. Check if the section flows naturally from the previous section (if provided)
        2. Verify the presence of effective transitions
        3. Ensure ideas progress logically within the section
        
        Return a JSON object with:
        - is_coherent: boolean indicating if the content flows well
        - feedback: string explaining any coherence issues found"""

        human_prompt = f"""Current Section Content:
{current_section}

{"Previous Section Content:" + previous_section if previous_section else "This is the first section."}"""

        try:
            structured_llm = self.llm.with_structured_output(CoherenceValidation)
            result = structured_llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt)
            ])
            return result.is_coherent, result.feedback
        except Exception as e:
            logger.warning(f"Failed to validate coherence via LLM: {str(e)}")
            # Fallback to basic transition check
            has_transition = bool(previous_section is None or 
                                any(transition in current_section.lower() 
                                    for transition in ['however', 'building on', 'following this', 'furthermore']))
            return has_transition, "Basic transition check only due to LLM error"

    def _repair_coherence(self, current_section: str, previous_section: Optional[str] = None, feedback: str = "") -> str:
        """Attempt to fix coherence issues in a section using LLM."""
        system_prompt = """You are an expert content editor. Your task is to improve the coherence of a section while preserving its core information.
        
        Guidelines:
        1. Add appropriate transitions based on the previous section
        2. Ensure logical flow of ideas
        3. Maintain the original information and citations
        4. Fix only coherence issues - do not change the core content
        5. Keep the same general length and depth
        
        Previous feedback on coherence issues: {feedback}"""

        human_prompt = f"""Current Section:
{current_section}

Previous Section Context:
{previous_section if previous_section else "This is the first section."}

Return the improved section with better coherence and transitions."""

        try:
            # Use raw LLM for content generation
            response = self.llm.invoke([
                SystemMessage(content=system_prompt.format(feedback=feedback)),
                HumanMessage(content=human_prompt)
            ])
            
            if isinstance(response.content, str) and response.content.strip():
                return response.content.strip()
            else:
                logger.warning("LLM returned invalid response during coherence repair")
                return current_section
        except Exception as e:
            logger.warning(f"Failed to repair coherence via LLM: {str(e)}")
            return current_section

    def _validate_seo(self, section_content: str, section_title: str, target_keywords: list[str]) -> tuple[bool, str, dict]:
        """Validate SEO optimization of a section using LLM."""
        system_prompt = """You are an SEO optimization expert. Analyze the content for SEO effectiveness.
        Consider:
        1. Keyword density (0.5-2.5% optimal)
        2. Keyword placement (early in section, in headings)
        3. LSI (Latent Semantic Indexing) keyword usage
        4. Readability and natural language flow
        5. Header tag optimization (H2, H3 usage)
        
        Return a JSON object with detailed analysis."""

        human_prompt = f"""Section Title: {section_title}
Content: {section_content}
Target Keywords: {', '.join(target_keywords)}

Analyze SEO optimization and return:
- is_optimized: boolean indicating if SEO is well optimized
- feedback: string explaining any SEO issues
- metrics: object containing:
  - keyword_density: float (percentage)
  - primary_keyword_position: int (first occurrence position)
  - lsi_keywords_found: list of related keywords found
  - readability_score: float (0-1)"""

        try:
            # Define expected response structure
            class SEOMetrics(BaseModel):
                keyword_density: float
                primary_keyword_position: int
                lsi_keywords_found: list[str]
                readability_score: float

            class SEOValidation(BaseModel):
                is_optimized: bool
                feedback: str
                metrics: SEOMetrics

            structured_llm = self.llm.with_structured_output(SEOValidation)
            result = structured_llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt)
            ])
            return result.is_optimized, result.feedback, result.metrics.dict()
        except Exception as e:
            logger.warning(f"Failed to validate SEO via LLM: {str(e)}")
            # Fallback to basic keyword check
            basic_density = sum(1 for kw in target_keywords if kw.lower() in section_content.lower()) / len(section_content.split()) * 100
            return basic_density > 0, "Basic keyword check only", {"keyword_density": basic_density}

    def _optimize_seo(self, section_content: str, section_title: str, target_keywords: list[str], seo_feedback: str) -> str:
        """Attempt to improve SEO optimization while maintaining content quality."""
        system_prompt = """You are an SEO optimization expert. Improve the content's SEO while maintaining its value and readability.
        
        Guidelines:
        1. Optimize keyword placement without keyword stuffing
        2. Add relevant LSI keywords naturally
        3. Improve header structure (H2, H3) if needed
        4. Maintain the original message and tone
        5. Keep content natural and reader-friendly
        
        Previous SEO feedback: {feedback}"""

        human_prompt = f"""Section Title: {section_title}
Content: {section_content}
Target Keywords: {', '.join(target_keywords)}

Return the SEO-optimized content while preserving the core message."""

        try:
            response = self.llm.invoke([
                SystemMessage(content=system_prompt.format(feedback=seo_feedback)),
                HumanMessage(content=human_prompt)
            ])
            
            if isinstance(response.content, str) and response.content.strip():
                return response.content.strip()
            else:
                logger.warning("LLM returned invalid response during SEO optimization")
                return section_content
        except Exception as e:
            logger.warning(f"Failed to optimize SEO via LLM: {str(e)}")
            return section_content

    def _is_similar_content(self, content1: str, content2: str) -> bool:
        """Check if two content blocks are semantically similar using LLM."""
        system_prompt = """You are a semantic similarity detector. Your task is to determine if two content blocks are semantically similar enough that they would create redundancy in a blog post.
        Consider:
        1. Core topic overlap
        2. Main concepts being discussed
        3. Overall intent and message
        4. Key points and examples used
        Do not consider minor variations in wording if the core meaning is the same.
        
        Return a JSON object with a single boolean field "is_similar"."""

        human_prompt = f"""Content Block 1:
{content1[:1000]}  # Limit size to avoid token limits

Content Block 2:
{content2[:1000]}  # Limit size to avoid token limits

Are these content blocks semantically similar enough that having both would create redundancy in a blog post?"""

        try:
            structured_llm = self.llm.with_structured_output(SimilarityCheck)
            result = structured_llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt)
            ])
            return result.is_similar
        except Exception as e:
            logger.warning(f"Failed to check content similarity via LLM, falling back to basic check: {str(e)}")
            # Fallback to basic similarity if LLM fails
            content1_words = set(content1.lower().split())
            content2_words = set(content2.lower().split())
            overlap = len(content1_words.intersection(content2_words))
            total = len(content1_words.union(content2_words))
            return overlap / total > 0.8 if total > 0 else False

    def _revise_chunk(self, blog_post: BlogPost, target_read_time: Optional[int] = None) -> BlogPost:
        """Add content validation before revision"""
        # Read system prompt
        with open(self.REVISE_CONTENT_SYSTEM_PROMPT, "r") as file:
            system_prompt = file.read()
        with open(self.REVISE_CONTENT_HUMAN_PROMPT, "r") as file:
            human_prompt_template = file.read()

        # Format sections
        formatted_sections = "\n".join(f"## {section.title}\n{section.content}" for section in blog_post.sections)

        # Format human prompt
        human_prompt = human_prompt_template.format(
            title=blog_post.title,
            sections=formatted_sections,
            target_read_time=target_read_time or "No specific target"
        )

        # Create structured output LLM
        structured_llm = self.llm.with_structured_output(BlogPost)

        # Revise content with better error handling and retries
        max_retries = 3
        for attempt in range(max_retries):
            try:
                revised_post = structured_llm.invoke([
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=human_prompt)
                ])
                
                # Enhanced validation
                if not revised_post or not revised_post.sections:
                    logger.error(f"LLM returned empty response (attempt {attempt + 1}/{max_retries})")
                    if attempt == max_retries - 1:
                        logger.error("Max retries reached, returning original blog post")
                        return blog_post
                    continue

                # Process sections
                seen_titles = []  # Changed to list to preserve titles for comparison
                seen_contents = []  # Track content to check for duplicates
                valid_sections = []
                prev_section_content = None
                
                for section in revised_post.sections:
                    # Basic validation first
                    if not section.title.strip() or not section.content.strip():
                        logger.warning(f"Skipping invalid section: {section}")
                        continue
                        
                    # Check for similar titles and content
                    is_duplicate = False
                    for idx, (seen_title, seen_content) in enumerate(zip(seen_titles, seen_contents)):
                        if self._is_similar_title(section.title, seen_title) or self._is_similar_content(section.content, seen_content):
                            logger.warning(f"Detected duplicate section: '{section.title}' is similar to '{seen_title}'")
                            is_duplicate = True
                            break
                    
                    if is_duplicate:
                        continue
                    
                    seen_titles.append(section.title)
                    seen_contents.append(section.content)
                    
                    # Enhanced citation validation
                    valid_citations = []
                    for citation in section.citations:
                        is_valid, formatted_citation = self._validate_citation(citation)
                        if is_valid:
                            valid_citations.append(formatted_citation)
                        else:
                            logger.warning(f"Invalid/unfixable citation in section '{section.title}': {citation}")
                    
                    # Update section with validated citations
                    section.citations = valid_citations
                    
                    # Validate coherence if we have a previous section
                    if prev_section_content is not None:
                        is_coherent, feedback = self._validate_coherence(section.content, prev_section_content)
                        if not is_coherent:
                            logger.warning(f"Coherence issues in section '{section.title}': {feedback}")
                            # Try to repair coherence
                            section.content = self._repair_coherence(section.content, prev_section_content, feedback)
                    
                    valid_sections.append(section)
                    prev_section_content = section.content

                # If we have any valid sections, return the revised post
                if valid_sections:
                    revised_post.sections = valid_sections
                    return revised_post
                else:
                    logger.error("No valid sections after processing")
                    if attempt == max_retries - 1:
                        return blog_post
                    continue

            except Exception as e:
                logger.error(f"Error during revision (attempt {attempt + 1}/{max_retries}): {str(e)}")
                if attempt == max_retries - 1:
                    logger.error("Max retries reached, returning original blog post")
                    return blog_post
                continue

        return blog_post

    def invoke(self, state: BlogWorkflowState) -> BlogWorkflowState:
        blog_post = state.get("blog_post")
        target_read_time = state.get("target_read_time")

        if blog_post:
            revised_blog_post = self.revise_content(blog_post, target_read_time)
            state["blog_post"] = revised_blog_post
            state["editor_suggestions_applied"] = True

        return state 