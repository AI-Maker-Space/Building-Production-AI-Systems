# 🚀 Building Production AI Systems  
**Create an end-to-end LLM App — No Coding Required!**  

Welcome to your hands-on journey to building production-ready AI systems.  
In this guide, you'll learn how to design, build, and deploy a complete **LLM application** — from backend to frontend — using tools like **Cursor**, **v0**, **GitHub**, and **Vercel**.

---

## 🌟 Overview
This workshop teaches you how to:
- Build an AI-powered app with **no manual coding**
- Use **GitFlow** for structured collaboration
- Test and deploy your app live on the web
- Understand how AI systems are structured in production

---

## 🧭 Step-by-Step Guide

### 📝 Example Prompts for Building the Diet App

#### 🔧 FastAPI Backend Prompts (Use in Cursor)

**Prompt 1: Initial Backend Setup**
```
Create me a super simple but pretty FastAPI LLM app (using GPT-4.0) with a simple UI HTML and run with uvicorn. 
It will be a Thanksgiving Diet app where I select if I am vegetarian/vegan/no restrictions 
and it returns a simple thanksgiving recipe I can use for dinner. User will insert the 
OPENAI_API_KEY in the UI. 
Use client = OpenAI(api_key=api_key). 
Make sure the libraries are compatible with Python 3.13. 
Use openai==1.51.0 and use compatible httpx==0.27.0.

```

**Prompt 2: Vercel Deployment Configuration**
```
Add files to deploy to Vercel. Make sure there is no proxies argument error, 
error caused by a dependency conflict between the OpenAI library and its 
underlying HTTP library (httpx) on Vercel. Note: the OpenAI SDK 1.x+ doesn't 
accept a proxies parameter directly.
```

---

#### 💅 Frontend Prompts (Use in v0.dev)

**Prompt 1: Initial Frontend Setup**
```
Create me a Next.js frontend for a Thanksgiving diet LLM app with an image of turkey, where I select if I am 
vegetarian/vegan/or no restrictions and it returns a simple thanksgiving recipe. User will 
insert the OPENAI_API_KEY and NEXT_PUBLIC_API_URL directly in the UI! See the backend 
attached. Make sure to use React 18.3.1 and frontend uses vaul 1.1.1 version. 
Use all the necessary files so I can deploy it to Vercel. Use openai==1.51.0.
```

**Prompt 2: Connect Frontend & Backend**
```
Connect backend and the frontend (frontend is created with v0 in frontend 
folder) so I can run them using commands: npm and uvicorn. Make sure that both 
frontend and backend use React 18.3.1 and frontend uses vaul 1.1.1 version. 
Finally, Make sure the PostCSS configuration is not missing the required Tailwind CSS plugin. 
```

---

### 🪴 STEP 1 — Create a GitHub Repository  
1. Log into your [GitHub](https://github.com/) account.  
2. Click **New Repository** → name your project (e.g., `my-llm-app`).  
3. Clone it locally using **SSH keys** *(see our October 1 VibeCoding session for setup details)*:  
   ```bash
   git clone git@github.com:your-username/my-llm-app.git
   ```

---

### 🧩 STEP 2 — Set Up GitFlow Rules  
Paste the GitFlow rules we defined in our previous session into your README or team wiki.  
These ensure smooth collaboration between main, develop, and feature branches.

Example:  
```bash
main     → production-ready code  
develop  → integration branch  
feature/ → new features  
hotfix/  → urgent fixes  
```

---

### ⚙️ STEP 3 — Create Backend in Cursor  
1. Open your GitHub repo inside **Cursor**.  
2. Use the “Create backend” command in Cursor.  
3. Generate endpoints for your app (e.g., `/api/analyze`, `/api/results`).  
4. Commit and push the backend to GitHub.

---

### 💅 STEP 4 — VibeCode Your Frontend (v0.dev)  
Use **v0** to visually design your frontend:  
- Add buttons, inputs, and chat windows  
- Connect them to backend endpoints later  
- Export when finished

---

### 🧠 STEP 5 — Create a Virtual Environment  
In your project folder, create and activate a Python virtual environment:  
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

### 🧭 STEP 6 — Download Frontend into Your Repo Folder  
Export your v0 project and move it into your local repo folder:  
```
/my-llm-app
  ├── main.py
  ├── frontend/
  └── README.md
```

---

### 🔗 STEP 7 — Connect Frontend with Backend in Cursor  
Prompt Cursor:  
> “Connect my frontend with my backend API endpoints so I can run the app with npm and uvicorn commands.”  

Cursor will automatically create the integration code for you.

---

### 🧪 STEP 8 — Test Your App  
Before pushing to GitHub, make sure everything runs smoothly:  
1. Run your backend locally:  
   ```bash
   uvicorn api:app --reload
   ```  
2. Run your frontend (depending on your setup):  
   ```bash
   npm run dev
   ```  
3. Test your app locally by visiting the URL (e.g., `http://localhost:3000`).  
4. Check if the frontend correctly calls your backend endpoints.  

✅ *Pro Tip:* Use the browser console or network tab to see if requests and responses are working as expected.

---

### 🧑‍💻 STEP 9 — Push to GitHub Following GitFlow Rules  
Commit and push your code:  
```bash
git add .
git commit -m "Add full LLM app"
git push origin develop
```
Then merge via Pull Request to `main` for production.

---

### ☁️ STEP 10 — Deploy App to Vercel  
1. Go to [vercel.com](https://vercel.com).  
2. Import your GitHub repo.  
3. Configure environment variables if needed.  
4. Click **Deploy** — your app will go live in minutes!

---

## 🎉 Congratulations!
You’ve just created a **production-ready AI system** — end-to-end — without writing code manually!  

💡 *Next step:* Add custom logic, authentication, or analytics for your MVP.  

---

### 🪄 Helpful Links
- [Cursor](https://cursor.sh) — AI-powered coding environment  
- [v0.dev](https://v0.dev) — visual frontend builder  
- [GitHub SSH Keys Setup Guide](https://docs.github.com/en/authentication/connecting-to-github-with-ssh)  
- [Vercel Deployment Docs](https://vercel.com/docs/deployments/overview)

---

**Created for the AI Makerspace “VibeCoding” series**  
by *Katerina Gawthorpe* ✨  
