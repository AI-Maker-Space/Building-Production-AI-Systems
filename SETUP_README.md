# 🚀 AI Engineer Development Setup Guide

## Part 1: GitHub Setup & Configuration

### 🍎 macOS Apple Silicon Setup

**Goal:** Install essential development tools on Apple Silicon Macs.
### 🔗 Additional Setup Resources
Visit the comprehensive setup instructions: [Interactive Dev Environment Repo](https://github.com/AI-Maker-Space/Interactive-Dev-Environment-for-AI-Engineers)


#### 1.1 Open Terminal
Press **⌘+Space**, type **Terminal**, and press Enter.

#### 1.2 Install Homebrew
Run the following command and follow the prompts:
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

#### 1.3 Update Homebrew
This may take a few minutes:
```bash
git -C /usr/local/Homebrew/Library/Taps/homebrew/homebrew-core fetch --unshallow
git -C /usr/local/Homebrew/Library/Taps/homebrew/homebrew-cask fetch
```

#### 1.4 Install Essential Tools
**Download utility:**
```bash
brew install wget
```

**Command-line developer tools:**
```bash
xcode-select --install
```

#### 1.5 Install Git - version control and Python
| Tool | Purpose | Command / Link |
|------|---------|----------------|
| Git | Version Control | `brew install git` |
| Python | Python code | `brew install python` |

### 🔐 GitHub SSH Setup

**What is SSH?** Secure Shell Protocol provides a secure communication channel over an unsecured network.

#### 1.6 Generate SSH Key Pair
Run this command (replace with your GitHub email):
```bash
ssh-keygen -o -t rsa -C "your_email@example.com"
```
- Save the file pair in the default location (`~/.ssh/id_rsa`)
- At the prompt, type in a secure passphrase (optional)

#### 1.7 Copy Your Public Key
**Choose your operating system:**

**🍎 macOS:**
```bash
pbcopy < ~/.ssh/id_rsa.pub
```

**🪟 Windows (WSL):**
```bash
clip.exe < ~/.ssh/id_rsa.pub
```

**🐧 Linux:**
```bash
xclip -sel c < ~/.ssh/id_rsa.pub
```

#### 1.8 Add Key to GitHub
1. Go to your **GitHub account**
2. Open **Settings**
3. Under **Access**, click **SSH and GPG keys** in the left sidebar
4. Click **New SSH Key**
5. Give it a name, paste your public key, and click **Add SSH Key**

#### 1.9 Test Your Setup
✅ **Done!** You can now use SSH with GitHub:
```bash
git clone git@github.com:username/repo.git
```

### 📋 Next GitHub Steps
- Create repository
- Discuss branch management
- Review Git workflow documentation

## Part 2: Git Branch Management with GitFlow

## 🌳 ASCII Branch Diagram

```text
 main    ───●───────────────●───────────────●────────────▶
             ▲               ▲
             │               │
         hotfix/…        release/1.2.0
                             │
 develop ──●─────●─────●─────●──────────────●───────────▶

```

### 📋 Complete GitFlow Workflow Guide

This demonstrates a **real-life GitFlow workflow** with clear steps, explanations, and commands.
Can be applied using Claude code.

#### 🏗️ Setup `main` and `develop`

**Clone your repository:**
```bash
# Clone your repository and navigate into it
git clone git@github.com:yourname/yourrepo.git   # Clones the repository using SSH
cd yourrepo                                      # Changes directory to your new repo
```

**Create the `develop` branch:**
```bash
git checkout -b develop
git push -u origin develop
```

#### 📦 Prepare a Release

**Create a release branch from `develop`:**
```bash
git checkout develop
git pull origin develop
git checkout -b release/1.2.0
git push -u origin release/1.2.0
```

**Only bug fixes and version updates:**
```bash
git add .
git commit -m "Fix login validation bug"
git push
```

####  🚢 Release to Production

**Merge the release branch into `main`:**
```bash
git checkout main
git pull origin main
git merge --no-ff release/1.2.0
git push origin main
```

**Tag the release:**
```bash
git tag -a v1.2.0 -m "Release version 1.2.0"
git push origin v1.2.0
```

#### 🔄 Merge Release Back into `develop`

**Keep bug fixes in sync:**
```bash
git checkout develop
git pull origin develop
git merge --no-ff release/1.2.0
git push origin develop
```

**Clean up:**
```bash
git branch -d release/1.2.0
git push origin --delete release/1.2.0
```

## Part 3: Cursor IDE Setup & Configuration

### 🔗 Additional Setup Resources
Visit the comprehensive setup instructions: [Interactive Dev Environment Repo](https://github.com/AI-Maker-Space/Interactive-Dev-Environment-for-AI-Engineers)

#### 3.1 Install Cursor & VS Code
| Tool | Purpose | Command / Link |
|------|---------|----------------|
| 📝 VS Code | Development Environment | [Download](https://code.visualstudio.com/) |
| 📝 Cursor | Development Environment | [Download](https://cursor.sh/) |

✅ **Your environment is ready to start developing!**

### ⚙️ Configure Cursor (or VS Code) Environment

**Goal:** Set up your development environment with essential extensions and configurations.

#### 3.2 Install Python and Jupyter Extensions
**For Python development:**

1. Click the **Extensions** tab
2. Type **"Python"** in the search bar
3. Click **Install** on the **Python** extension
4. Type **"Jupyter"** in the search bar
5. Click **Install** on the **Microsoft Jupyter Notebook** extension

✅ **Your development environment is configured and ready to code!**

### 📋 Next Steps for Cursor Setup
- Configure GitHub integration in settings
- add cursor.md rule!
- Clone repository from GitHub via Cursor
- Set up project-specific configurations

## Part 4: Building Your First FastAPI Application WITH GitFlow

### 🐍 Python Virtual Environments

**Goal:** Create an isolated Python environment for your project.

#### 4.1 Create a Virtual Environment
```bash
python3 -m venv myenv
```

#### 4.2 Activate the Environment
**🍎 macOS/Linux:**
```bash
source myenv/bin/activate
```

**🪟 Windows:**
```bash
myenv\Scripts\activate
```

#### 4.3 Deactivate When Done
```bash
deactivate
```

### How FastAPI Handles a GET Request

When you start FastAPI with:

```bash
uvicorn main:app --reload
```

…and visit `http://127.0.0.1:8000/`, here's what happens:

1. **Browser sends GET request** → `GET /`
2. FastAPI looks for a matching path operation (`@app.get("/")`)
3. It finds your function:
   ```python
   def read_root():
       return {"message": "Hello, FastAPI is running locally!"}
   ```
4. FastAPI **calls** `read_root()` (just like any Python function).
5. The function returns a Python dictionary.
6. FastAPI **automatically converts the dictionary into JSON**.
7. The JSON is sent back as the response to the browser.

---

### Summary
👉 FastAPI runs your function `read_root()` and uses its return value as the response.
