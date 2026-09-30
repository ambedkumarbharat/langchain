# LangChain Chain Lab

An interactive Streamlit project for learning how common LangChain patterns work. It shows each stage of a chain as it runs and includes standalone command-line examples, runnable examples, and mini projects.

## Included patterns

| Pattern | What it demonstrates |
| --- | --- |
| Simple chain | Prompt -> LLM -> output parser |
| Sequential chain | The output of one chain becomes input for the next |
| Parallel chain | Independent notes and quiz branches run, then merge |
| Conditional chain | Feedback is classified and routed to a positive or negative response |

## Runnables

Examples for learning LangChain's runnable interface:

| Runnable | What it demonstrates |
| --- | --- |
| Basic runnable | The core runnable interface and execution flow |
| Chained runnable | Combining multiple runnables with the pipe operator |
| Conditional runnable | Routing input based on runtime conditions |
| Parallel runnable | Running independent tasks and combining their results |
| Custom runnable | Building reusable runnable components |

Update the table and commands below as you add your actual runnable files.

## Mini projects

Small, practical projects for applying the concepts learned in the examples:

| Mini project | Focus |
| --- | --- |
| AI quiz generator | Generate quizzes from a topic or document |
| Sentiment analyzer | Classify and explain the sentiment of text |
| Document summarizer | Summarize long documents |
| Multi-turn chatbot | Build a conversational assistant with memory |
| RAG application | Answer questions using a custom knowledge base |

## Project structure

```text
.
├── streamlit_chain.py          # Interactive Streamlit application
├── chain/
│   ├── simple_chain.py         # CLI simple-chain example
│   ├── sequential_chain.py     # CLI sequential-chain example
│   ├── parallel_chain.py       # CLI parallel-chain example
│   └── conditional_chain.py    # CLI conditional-chain example
├── runnables/                  # Runnable examples
├── mini_projects/              # Small end-to-end applications
├── requirements.txt
└── .env.example                # Safe environment-variable template
```

## Setup

1. Create and activate a virtual environment.

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

2. Install the dependencies.

   ```powershell
   pip install -r requirements.txt
   ```

3. Create your local environment file from the template.

   ```powershell
   Copy-Item .env.example .env
   ```

4. Put your own credentials in `.env`:

   ```dotenv
   GROQ_API_KEY=your_groq_api_key
   GOOGLE_API_KEY=your_google_ai_api_key
   OPENROUTER_API_KEY=your_openrouter_api_key
   ```

## Run the Streamlit app

```powershell
streamlit run streamlit_chain.py
```

Open the local URL shown in the terminal, normally `http://localhost:8501`.

## Run individual examples

### Chain examples

```powershell
python chain/simple_chain.py
python chain/sequential_chain.py
python chain/parallel_chain.py
python chain/conditional_chain.py
```

### Runnable examples

Add commands here as runnable files are created:

```powershell
python runnables/<runnable_name>.py
```

### Mini projects

Run each mini project using the command documented in its own folder. For example:

```powershell
python mini_projects/<project_name>/app.py
```

## Security

`.env`, `.venv/`, caches, logs, and other local-only files are excluded through `.gitignore`. Do not put API keys directly in source code or commit your `.env` file.

## Adding new content

### Add a runnable

1. Add the example inside `runnables/`.
2. Include a short docstring or README explaining the concept.
3. Add its run command to this README.
4. Test it locally before committing.

### Add a mini project

1. Create a clearly named folder inside `mini_projects/`.
2. Include the source code, a short README, and setup/run instructions.
3. Document required environment variables without including real keys.
4. Add the project to the mini-projects table above.

Then commit and push your changes:

```powershell
git add chain runnables mini_projects README.md
git commit -m "Add runnable or mini project"
git push
```

## Learning resources

- [LangChain Documentation](https://python.langchain.com/)
- [LangChain GitHub Repository](https://github.com/langchain-ai/langchain)
- [LangChain Expression Language](https://python.langchain.com/docs/concepts/lcel/)

## License

Add the project's license information here when a license is selected.
