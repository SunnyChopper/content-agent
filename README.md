# Content Agent
**Goal**: Create an LLM agent system (by leveraging LangGraph) that can write high-quality, accurate and engaging content. The system uses multiple specialized agents working together to research, write, review and edit content.

## Atomic Agents
Decomposing a multi-agent system for writing high-quality technical blog posts with a target read time can be effectively achieved using the Atomic Design pattern applied to LangGraph agents. Here's a possible breakdown:

1. **Research Agent**: This agent is responsible for gathering information from the web to build a knowledge base. It uses DuckDuckGo Search to find relevant information and Wikipedia Search to find detailed information. (MVP+: Add keyword search for historical search volume to boost SEO)
2. **Writing Agent**: This agent is responsible for creating the outline to match the target read time and then writing the content section-by-section. If any business goals are provided, it will integrate them into the content in a subtle, convincing manner. (MVP+: Integrate the top keywords from the research agent into the content for SEO)
3. **Review Agent**: This agent is responsible for reviewing the content and ensuring it is accurate and engaging by comparing it to the knowledge base and web search results provided by the Research Agent. It flags any errors or areas that need improvement based on contradictory information. (MVP+: Detect knowledge gaps in the knowledge base and web search results, which triggers the Research Agent to gather more information)
4. **Editing Agent**: This agent is responsible for editing the content to ensure it is engaging and reads well. It also inserts hyperlinks into the content using the URLs from the search tool that were used by the Research Agent. Additionally, it inserts search queries for images and videos to insert into the content in post manual editing. (MVP+: Automatically search and insert images and videos into the content using a smart vision model)
