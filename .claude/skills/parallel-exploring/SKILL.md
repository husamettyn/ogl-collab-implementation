---
name: parallel-exploring
description: Use this skill whenever the user wants to explore, investigate, or understand the codebase — especially broad questions like "how does X work", "find all the places where", "what's the architecture", "search the repo for", or any question that spans multiple files or components. Also use it when the user says "explore", "investigate", "look into", "dig into", "scan the codebase", or asks about project structure, data flow, or how components fit together. This skill decomposes the question, spawns parallel search agents, and synthesizes findings into a clear report with file paths and line numbers.
---

# Parallel Exploring

When a user asks a broad or multi-faceted question about the codebase, break it into focused sub-questions and explore them in parallel using specialized search agents. Then synthesize the results into a single clear report.

## When to use this skill

This skill fits any codebase exploration that can't be answered by reading one or two files. Signs you should use it:

- The question touches multiple directories, modules, or concerns
- The user asks "how does X work" where X is a system, pipeline, or flow
- The user wants to find all places where a pattern, concept, or dependency appears
- The user asks about architecture, structure, or how components connect
- The user says "explore the codebase" or similar

Skip this skill for single-file lookups, simple grep searches, or questions answerable from one known location.

## Step 1: Decompose the question

Break the user's question into 2–4 independent sub-questions. Each sub-question should:

- Be answerable by searching/grepping/reading files — no need for the others
- Cover a distinct area (different directories, different concerns, different layers)
- Together cover the full scope of the original question

If the question is inherently sequential (B depends on knowing A), do a quick scout first, then spawn the dependent searches.

## Step 2: Spawn parallel Explore agents

For each sub-question, spawn an Explore agent using the Agent tool. Run all independent ones in parallel — send them in a single message.

Each agent prompt should include:
- The exact sub-question to answer
- Suggested search breadth ("quick" for narrow, "medium" for moderate, "very thorough" for deep)
- Specific file patterns or directories to focus on (when known)
- Instructions to cite file paths and line numbers for every finding

Example:

```
Agent(
  subagent_type: "Explore",
  description: "Search auth middleware",
  prompt: "Find all authentication middleware in this project. Look in src/middleware/ and src/auth/. Search for patterns like 'auth', 'token', 'session', 'login'. Report every file path and line number where auth logic is defined. Breadth: medium."
)
```

Use `run_in_background: true` when you have more than 2 agents, so they truly run in parallel.

## Step 3: Synthesize the findings

Once all agents return, combine their findings into a report with this structure:

```
# [Topic]: Exploration Results

## Summary
[2–3 sentence overview of what was found, connecting the dots across sub-questions]

## [Sub-question 1 heading]
[Findings from agent 1, organized clearly, with file paths and line numbers cited]

## [Sub-question 2 heading]
[Findings from agent 2, same format]

...
```

Principles for the synthesis:
- **Connect the dots**: Don't just paste agent outputs. Explain how findings relate to each other.
- **Cite everything**: Every claim about the code should reference a file path and line number.
- **Be concise**: The report should be skimmable. Put the most important findings first.
- **Flag gaps**: If a sub-question wasn't fully answered, say so and suggest next steps.
