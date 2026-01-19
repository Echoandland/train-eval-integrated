# Security Guidelines

## Sensitive Information

This repository is designed to be public-ready. Before pushing to a public repository, ensure:

### API Keys and Tokens

**Never commit these to the repository:**

- `DASHSCOPE_API_KEY` - Alibaba Cloud API key
- `OPENAI_API_KEY` - OpenAI API key
- `HF_TOKEN` / `hf_token` - Hugging Face access token
- Any other API keys or tokens

**Best practices:**

1. Store secrets in environment variables
2. Use the `.env` file (which is gitignored)
3. Copy `env.example` to `.env` and fill in your values

### Paths and Personal Information

Avoid committing:

- Absolute paths containing usernames (e.g., `/home/username/...`)
- Personal email addresses
- Organization-specific cluster configurations

### Checking for Secrets

Before committing, run these checks:

```bash
# Check for potential API keys
grep -r "sk-[A-Za-z0-9]" --include="*.py" --include="*.sh" .
grep -r "hf_[A-Za-z0-9]" --include="*.py" --include="*.sh" .

# Check for hardcoded paths
grep -r "/home/" --include="*.py" --include="*.sh" .
grep -r "/Users/" --include="*.py" --include="*.sh" .

# Check for email patterns
grep -rE "[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}" --include="*.py" --include="*.sh" .
```

### Pre-commit Hook (Recommended)

Add a pre-commit hook to catch secrets before they're committed:

```bash
#!/bin/bash
# .git/hooks/pre-commit

if git diff --cached --name-only | xargs grep -l "sk-[A-Za-z0-9]\{20,\}" 2>/dev/null; then
    echo "ERROR: Potential API key detected in staged files"
    exit 1
fi

if git diff --cached --name-only | xargs grep -l "hf_[A-Za-z0-9]\{20,\}" 2>/dev/null; then
    echo "ERROR: Potential HuggingFace token detected in staged files"
    exit 1
fi
```

## Reporting Security Issues

If you discover a security vulnerability, please report it privately rather than opening a public issue.
