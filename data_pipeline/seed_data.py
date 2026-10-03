"""Hand-written seed prompts for MicDrop.

These are clean target prompts (the "output" side of training examples),
written to match the actual distribution of MicDrop usage: short spoken
requests to a coding/assistant agent, heavy on technical terms, file paths,
numbers/versions, and multi-part requests.

A small number of DOWIS-style task phrasings are included as general
conversational prompts. The noise engine (noisy_inputs.py) turns each seed
into 1-3 realistic spoken/Whisper-like inputs; a portion of seeds is also
kept as clean (non-noisy) examples so the model does not learn that every
input needs aggressive rewriting.

No dataset statistics here are derived from external sources; every line is
authored for MicDrop. License: original content written for this project.
"""

from __future__ import annotations

# (clean_prompt, category) tuples. Categories follow the project taxonomy:
#   programming, technical_correction, research, creative, questions,
#   multi_part, numbers_versions, file_paths, commands, long_requests

SEED_PROMPTS: list[tuple[str, str]] = [
    # ------------------------------------------------------- programming ----
    ("Write a Python script that reads a CSV file and calculates the average for each column.", "programming"),
    ("Create a FastAPI backend with PostgreSQL authentication.", "programming"),
    ("Write a Python function that flattens a nested dictionary into a single-level dictionary with dotted keys.", "programming"),
    ("Set up a PyTorch training loop with gradient accumulation and mixed precision.", "programming"),
    ("Create a React dashboard for tracking expenses, using a dark theme and displaying charts.", "programming"),
    ("Write a TypeScript utility that retries a fetch call with exponential backoff.", "programming"),
    ("Build a Next.js page that loads posts from the JSONPlaceholder API and paginates them.", "programming"),
    ("Write a Node.js script that renames all files in a folder to lowercase with underscores.", "programming"),
    ("Create a Python class implementing a rate limiter using a sliding window.", "programming"),
    ("Write a SQL query that finds the top ten customers by total order value.", "programming"),
    ("Write a regular expression that validates an ISO 8601 timestamp.", "programming"),
    ("Create a Docker Compose file for a Django app with Redis and Postgres.", "programming"),
    ("Write a Python script that watches a directory and uploads new files to S3.", "programming"),
    ("Implement a binary search tree in Rust with insert, delete, and search.", "programming"),
    ("Write a JavaScript debounce function and explain the parameters.", "programming"),
    ("Create a FastAPI endpoint that accepts a file upload and stores it on disk.", "programming"),
    ("Write a Go worker pool with graceful shutdown on SIGTERM.", "programming"),
    ("Write a Python decorator that times function execution and logs slow calls.", "programming"),
    ("Create a React hook that syncs state to local storage.", "programming"),
    ("Write a script that converts a JSON file to YAML.", "programming"),
    ("Implement an LRU cache in Python using ordered dict.", "programming"),
    ("Write a Django migration that adds an index on the created at column.", "programming"),
    ("Create a CSS grid layout with a responsive sidebar that collapses on mobile.", "programming"),
    ("Write a Python context manager that temporarily changes the working directory.", "programming"),
    ("Parse a log file and count requests per IP address using Python.", "programming"),
    ("Write a bash one-liner that finds all files larger than one hundred megabytes.", "programming"),
    ("Create a Kubernetes deployment manifest for a stateless API with three replicas.", "programming"),
    ("Write a GitHub Actions workflow that runs pytest on every push.", "programming"),
    ("Write a Python script that sends a Slack message when a test fails.", "programming"),
    ("Create a SQLAlchemy model with a one to many relationship between users and posts.", "programming"),

    # ----------------------------------------------- technical_correction ----
    ("What is the difference between malloc and calloc in C?", "technical_correction"),
    ("How do I train a LoRA adapter on Qwen with PyTorch?", "technical_correction"),
    ("Show me how to fine-tune a Hugging Face model on a custom dataset.", "technical_correction"),
    ("How do I pass CUDA device visibility to a Docker container?", "technical_correction"),
    ("Write a LangChain agent that can query a PostgreSQL database.", "technical_correction"),
    ("How do I convert a pandas data frame column from string to datetime?", "technical_correction"),
    ("What is the fastest way to embed one million documents for a RAG pipeline?", "technical_correction"),
    ("How do I run Whisper on a GPU with only four gigabytes of VRAM?", "technical_correction"),
    ("Set up a vector database with pgvector for semantic search.", "technical_correction"),
    ("How do I quantize a large language model to four bits with bitsandbytes?", "technical_correction"),
    ("Write a TensorFlow model that classifies images from the CIFAR ten dataset.", "technical_correction"),
    ("How do I profile memory usage of a PyTorch model during training?", "technical_correction"),
    ("Create a retrieval augmented generation pipeline with local embeddings.", "technical_correction"),
    ("How do I serve a Hugging Face model with FastAPI in Docker?", "technical_correction"),
    ("Write a script that benchmarks inference speed of two transformer models.", "technical_correction"),
    ("How do I shard a large language model across two GPUs with accelerate?", "technical_correction"),
    ("Configure OIDC single sign-on for a Grafana instance.", "technical_correction"),
    ("How do I stream OpenAI API responses server-sent events style in Express?", "technical_correction"),
    ("Write a Python script that counts tokens with the OpenAI tiktoken library.", "technical_correction"),
    ("How do I enable SSH key authentication and disable passwords on Fedora?", "technical_correction"),
    ("Set up a cron job that backs up a Postgres database every night at two a.m.", "technical_correction"),
    ("How do I create a systemd user service for a Python script on Fedora?", "technical_correction"),
    ("Write an awk command that sums the second column of a CSV.", "technical_correction"),
    ("How do I set up nginx as a reverse proxy for a Node app with TLS?", "technical_correction"),
    ("Create a Makefile target that builds, tests, and lints a Python project.", "technical_correction"),

    # ---------------------------------------------------- numbers_versions ----
    ("Migrate our project from Python three point nine to three point twelve.", "numbers_versions"),
    ("Upgrade the app to Next.js fourteen and fix any breaking changes.", "numbers_versions"),
    ("Install CUDA twelve point one on Fedora with the NVIDIA driver.", "numbers_versions"),
    ("Write a config that pins PyTorch two point one and transformers four point thirty-six.", "numbers_versions"),
    ("Create fifty test users in the database with random names.", "numbers_versions"),
    ("Generate a report of the top one hundred slowest API endpoints.", "numbers_versions"),
    ("Set the rate limit to two hundred requests per minute per API key.", "numbers_versions"),
    ("Retry failed jobs up to five times with a delay of thirty seconds.", "numbers_versions"),
    ("Reserve two gigabytes of shared memory for the Postgres container.", "numbers_versions"),
    ("Run the benchmark on batch sizes sixteen, thirty-two, and sixty-four.", "numbers_versions"),
    ("Add support for Node twenty-two and drop Node eighteen.", "numbers_versions"),
    ("Limit the Docker log size to ten megabytes with three rotated files.", "numbers_versions"),
    ("Schedule the job to run every six hours on weekdays.", "numbers_versions"),
    ("Give the service one CPU and five hundred twelve megabytes of memory.", "numbers_versions"),
    ("Write tests for the pagination function covering page sizes one, ten, and fifty.", "numbers_versions"),

    # ---------------------------------------------------------- file_paths ----
    ("Copy all JSON config files from /etc to my home directory.", "file_paths"),
    ("Write a script that moves files older than seven days from /tmp to /var/archive.", "file_paths"),
    ("Find every TODO comment under src slash components and list them.", "file_paths"),
    ("Back up the database dump at /backups/db.sql to S3 under the micdrop prefix.", "file_paths"),
    ("Read the config at ~/.config/micdrop/settings.json and pretty-print it.", "file_paths"),
    ("Search the log file /var/log/nginx/error.log for five hundred errors.", "file_paths"),
    ("Create a symbolic link from ~/projects/micdrop to the latest release folder.", "file_paths"),
    ("Sync my local folder with the remote server using rsync excluding node modules.", "file_paths"),
    ("Extract the archive at /opt/releases/app.tar.gz and run the install script.", "file_paths"),
    ("Write a script that lists the ten largest files under /var with their sizes.", "file_paths"),

    # ---------------------------------------------------------- multi_part ----
    ("Create a REST API with authentication, rate limiting, and OpenAPI docs.", "multi_part"),
    ("Set up a git repository with a pre-commit hook that formats and lints.", "multi_part"),
    ("Build a CLI that can add, list, and complete tasks and stores them in a JSON file.", "multi_part"),
    ("Write a script that downloads the dataset, preprocesses it, and trains a model.", "multi_part"),
    ("Create a React form with validation, error messages, and a loading state.", "multi_part"),
    ("Set up logging with rotation, add metrics, and alert on errors over five percent.", "multi_part"),
    ("Refactor the service layer, add type hints, and write unit tests for it.", "multi_part"),
    ("Create a table for users, one for sessions, and add a foreign key between them.", "multi_part"),
    ("Fork the repo, add a CI pipeline, and open a pull request with the changes.", "multi_part"),
    ("Write an essay outline, then a first draft, then a checklist for editing.", "multi_part"),
    ("Dockerize the app, push the image to the registry, and deploy it to the cluster.", "multi_part"),
    ("Add a dark mode toggle, persist the choice, and apply it without a page reload.", "multi_part"),

    # ------------------------------------------------------------ commands ----
    ("Run the tests with pytest and show only failures.", "commands"),
    ("Start the dev server on port three thousand.", "commands"),
    ("Rebuild the Docker image without cache.", "commands"),
    ("List all branches that have not been merged into main.", "commands"),
    ("Show the git diff for the last commit.", "commands"),
    ("Kill the process listening on port eight thousand.", "commands"),
    ("Reset the database and run all migrations from scratch.", "commands"),
    ("Check disk usage per folder in the current directory.", "commands"),
    ("Show the last fifty lines of the app log and follow it.", "commands"),
    ("Create a new Python virtual environment and install the requirements.", "commands"),

    # ------------------------------------------------------------- research ----
    ("Summarize the tradeoffs between SQL and NoSQL databases for analytics.", "research"),
    ("What are the current best practices for structuring a large React codebase?", "research"),
    ("Compare LoRA and full fine-tuning for small language models.", "research"),
    ("Research whether WebGPU is ready for production use in 2026.", "research"),
    ("Find the main differences between systemd timers and cron on Fedora.", "research"),
    ("What are the most reliable metrics for measuring retrieval quality in RAG?", "research"),
    ("Compare FastAPI and Flask for a small machine learning API.", "research"),
    ("What is the current state of local speech recognition on Linux?", "research"),
    ("Summarize the options for running large language models locally.", "research"),
    ("Explain how vision transformers differ from convolutional networks.", "research"),

    # ------------------------------------------------------------- creative ----
    ("Write a short product description for a mechanical keyboard with hot-swappable switches.", "creative"),
    ("Draft a friendly changelog entry for version two point zero.", "creative"),
    ("Write a README introduction for a voice-controlled terminal tool.", "creative"),
    ("Create three name ideas for a podcast about machine learning.", "creative"),
    ("Write a polite email asking a contributor to rebase their pull request.", "creative"),
    ("Draft release notes for a minor update with two bug fixes.", "creative"),
    ("Write a blog post outline about building voice interfaces on Linux.", "creative"),
    ("Create a short onboarding message for new users of the app.", "creative"),

    # ------------------------------------------------------------ questions ----
    ("Why does my React component render twice in development?", "questions"),
    ("How does garbage collection work in Python?", "questions"),
    ("What does the tilde do in front of a path on Linux?", "questions"),
    ("Why is my PyTorch loss not decreasing during training?", "questions"),
    ("How do environment variables differ between systemd and bash?", "questions"),
    ("What is the difference between a merge and a rebase in git?", "questions"),
    ("Why does my Docker container not see the GPU?", "questions"),
    ("How do I know which process is using a port on Fedora?", "questions"),
    ("What is the difference between append and extend in Python lists?", "questions"),
    ("Why is my Wayland clipboard not shared between apps?", "questions"),
    ("How does LoRA reduce memory usage during fine-tuning?", "questions"),
    ("What is the difference between async and multithreading in Python?", "questions"),

    # --------------------------------------------------------- long_requests ----
    ("Write a Python module that provides a command-line interface with subcommands for starting, stopping, and checking the status of a background service, including colored output and a config file.", "long_requests"),
    ("Create a data pipeline that reads events from Kafka, validates them against a JSON schema, writes the valid ones to Parquet, and publishes a daily summary report.", "long_requests"),
    ("Build a monitoring dashboard that shows CPU, memory, and GPU usage for the training servers and sends an alert when a GPU is idle for more than an hour.", "long_requests"),
    ("Write a tutorial that explains how to fine-tune a small language model on a custom dataset, covering data preparation, training, evaluation, and deployment.", "long_requests"),
    ("Refactor the authentication module to support OAuth two, API keys, and session cookies at the same time, keeping backward compatibility with existing clients.", "long_requests"),
    ("Create a CI pipeline that lints, type-checks, tests, builds the Docker image, and deploys to staging when all checks pass on main.", "long_requests"),
    ("Write a script that compares two JSON files, reports added, removed, and changed keys recursively, and exits with a nonzero code on differences.", "long_requests"),
    ("Design the database schema for a bookmarking service with users, tags, folders, and full-text search, and write the migration files.", "long_requests"),

    # ------------------------------------ general conversational (DOWIS-style) ----
    ("Write out what is being said in this audio.", "general"),
    ("Summarize what was said in this meeting.", "general"),
    ("Translate what the other person said into German.", "general"),
    ("Give me a short summary of this article.", "general"),
    ("Turn this text into natural spoken English.", "general"),
    ("Make the summary as short as possible.", "general"),
    ("Please write the translation in a formal style.", "general"),
    ("Listen to this recording and write down every word.", "general"),
    ("What did they say about the deadline in the call?", "general"),
    ("Rewrite this paragraph so it sounds informal.", "general"),
    ("Give me the key points of this talk in bullet points.", "general"),
    ("Convert the first chapter of this document into speech.", "general"),
]

assert SEED_PROMPTS, "seed list must not be empty"
