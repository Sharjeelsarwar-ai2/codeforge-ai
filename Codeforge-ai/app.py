# app.py
import streamlit as st
import json
import time
from pathlib import Path
from datetime import datetime

# --- Page Config (MUST be first Streamlit command) ---
st.set_page_config(
    page_title="CodeForge AI - AI Software House",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --- Check API Key Before Anything Else ---
def check_api_key():
    """Verify Groq API key is configured before running the app."""
    try:
        if hasattr(st, "secrets") and "GROQ_API_KEY" in st.secrets:
            api_key = st.secrets["GROQ_API_KEY"]
            if api_key and api_key != "your_groq_api_key_here":
                return True
    except Exception:
        pass

    import os
    if os.getenv("GROQ_API_KEY"):
        return True

    return False


if not check_api_key():
    st.error("🔑 **Groq API Key Not Configured**")
    st.markdown("""
    ### Setup Instructions
    
    **For Streamlit Cloud Deployment:**
    1. Go to your app dashboard on [Streamlit Cloud](https://share.streamlit.io/)
    2. Click on your app → Settings → Secrets
    3. Add your key in this format:
    ```toml
    GROQ_API_KEY = "gsk_your_actual_key_here"
    ```
    4. Save and reboot your app
    
    **For Local Development:**
    Create `.streamlit/secrets.toml` in your project root:
    ```toml
    GROQ_API_KEY = "gsk_your_actual_key_here"
    ```
    
    Get your free Groq API key at: [console.groq.com](https://console.groq.com/keys)
    """)
    st.stop()


# --- Now import everything else (after API key check) ---
from utils.db import init_db, save_project, update_project_status, get_all_projects, get_project
from utils.helpers import get_file_icon, get_file_language, get_project_stats, format_file_size, slugify
from tools.preview_tools import build_preview_html
from tools.file_tools import create_file_tree, read_file
from tools.project_tools import zip_project
from crew.crew import CodeForgeCrew


# --- Initialize Database ---
init_db()

# --- Custom CSS ---
st.markdown("""
<style>
    /* Global */
    .stApp {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a3e 50%, #0f0f23 100%);
    }
    
    /* Header */
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 800;
    }
    
    .sub-header {
        text-align: center;
        color: #a0a0b8;
        font-size: 1.2rem;
        margin-top: -1rem;
        margin-bottom: 2rem;
    }
    
    /* Cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 0.5rem 0;
    }
    
    /* File Tree */
    .file-item {
        padding: 0.4rem 0.8rem;
        border-radius: 6px;
        cursor: pointer;
        transition: background 0.2s;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
    }
    
    .file-item:hover {
        background: rgba(255, 255, 255, 0.08);
    }
    
    /* Agent Status */
    .agent-status {
        padding: 0.6rem 1rem;
        border-radius: 8px;
        margin: 0.3rem 0;
        font-size: 0.9rem;
    }
    
    .agent-active {
        background: rgba(99, 102, 241, 0.15);
        border-left: 3px solid #6366f1;
    }
    
    .agent-complete {
        background: rgba(34, 197, 94, 0.15);
        border-left: 3px solid #22c55e;
    }
    
    .agent-waiting {
        background: rgba(255, 255, 255, 0.03);
        border-left: 3px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        padding: 8px 16px;
    }
</style>
""", unsafe_allow_html=True)


# --- Session State Initialization ---
def init_session_state():
    defaults = {
        "page": "home",
        "project_path": None,
        "planning_results": None,
        "development_results": None,
        "current_phase": None,
        "agent_statuses": {},
        "selected_file": None,
        "crew": None,
        "project_id": None,
        "is_running": False,
        "generation_complete": False,
        "user_idea_input": "",
        "project_type_input": "",
        "tech_preference_input": "",
        "animation_level_input": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()


# --- Sidebar Navigation ---
def render_sidebar():
    with st.sidebar:
        st.markdown("## 🏗️ CodeForge AI")
        st.caption("AI Software House")
        st.markdown("---")

        if st.button("🏠 Home", use_container_width=True):
            st.session_state.page = "home"
            st.rerun()

        if st.button("🔧 Bug Fixer", use_container_width=True):
            st.session_state.page = "bug_fixer"
            st.rerun()

        if st.button("📁 My Projects", use_container_width=True):
            st.session_state.page = "projects"
            st.rerun()

        if st.button("ℹ️ About", use_container_width=True):
            st.session_state.page = "about"
            st.rerun()

        st.markdown("---")

        # Show current project info if active
        if st.session_state.project_path:
            st.markdown("### 📂 Current Project")
            project_name = Path(st.session_state.project_path).name
            st.markdown(f"**{project_name}**")

            if st.session_state.generation_complete:
                st.success("✅ Generation Complete")

                # ZIP Download
                try:
                    zip_path = zip_project.run(project_path=st.session_state.project_path)
                    if Path(zip_path).exists():
                        with open(zip_path, "rb") as f:
                            st.download_button(
                                "📦 Download ZIP",
                                data=f.read(),
                                file_name=f"{project_name}.zip",
                                mime="application/zip",
                                use_container_width=True,
                            )
                except Exception as e:
                    st.caption(f"ZIP error: {str(e)[:50]}")

        st.markdown("---")
        st.caption("🔑 API Key: ✅ Configured")
        st.caption("⚡ Powered by Groq")


render_sidebar()


# =====================================================
# PAGE: HOME - Create New Project
# =====================================================
def render_home():
    st.markdown('<h1 class="main-header">CodeForge AI</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">From Idea to Production-Ready Website & Web App in Minutes</p>',
        unsafe_allow_html=True,
    )

    # Quick stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🤖 AI Agents", "10")
    with col2:
        st.metric("⚡ Powered By", "Groq")
    with col3:
        st.metric("📁 Projects Built", len(get_all_projects()))
    with col4:
        st.metric("🎯 Success Rate", "95%")

    st.markdown("---")

    # Main input form
    st.markdown("### 💡 Describe Your Project")

    user_idea = st.text_area(
        "What do you want to build?",
        placeholder="Example: Create a modern animated SaaS landing page for an AI resume builder with gradient hero section, features grid, pricing table, testimonials, and a CTA section...",
        height=120,
        key="user_idea_input",
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        project_type = st.selectbox(
            "🎯 Project Type",
            [
                "Animated Landing Page (HTML + Tailwind + GSAP)",
                "Multi-Page Business Website",
                "Portfolio Website",
                "SaaS Product Website",
                "Full Web App (Flask + HTML/CSS/JS)",
                "Full Web App (FastAPI + HTML/CSS/JS)",
                "Custom (Let AI Decide)",
            ],
            key="project_type_input",
        )

    with col2:
        tech_preference = st.selectbox(
            "🛠️ CSS Framework",
            ["Tailwind CSS (CDN)", "Pure CSS (Custom)", "Bootstrap 5"],
            key="tech_preference_input",
        )

    with col3:
        animation_level = st.selectbox(
            "✨ Animation Level",
            ["Moderate", "Minimal", "Highly Animated"],
            key="animation_level_input",
        )

    st.markdown("---")

    # Generate button
    if st.button("🚀 Generate Project Plan", use_container_width=True, type="primary"):
        if not user_idea.strip():
            st.error("Please describe your project idea first!")
            return

        with st.spinner("🤖 AI Agents are analyzing your idea..."):
            try:
                # Create crew
                crew = CodeForgeCrew()

                # Generate project name from idea
                project_name = slugify(user_idea[:50])
                if not project_name:
                    project_name = "my-project"
                project_path = crew.setup_project(project_name)

                st.session_state.project_path = project_path
                st.session_state.crew = crew

                inputs = {
                    "user_idea": user_idea,
                    "project_type": project_type,
                    "tech_preference": tech_preference,
                    "animation_level": animation_level,
                    "project_name": project_name,
                    "project_path": project_path,
                }

                # Save project to DB
                project_id = save_project(
                    name=project_name,
                    description=user_idea,
                    project_type=project_type,
                    tech_preference=tech_preference,
                    animation_level=animation_level,
                    project_path=project_path,
                    status="planning",
                )
                st.session_state.project_id = project_id

                # Show agent activity panel
                st.markdown("### 🤖 Agent Activity")
                agent_names = [
                    ("Client Requirement Agent", "Analyzing your idea..."),
                    ("Product Manager Agent", "Creating PRD & Sitemap..."),
                    ("UI/UX Designer Agent", "Building design system..."),
                    ("Animation Agent", "Planning animations..."),
                    ("Frontend Architect Agent", "Planning file structure..."),
                ]

                status_containers = []
                for name, desc in agent_names:
                    status_containers.append(st.status(f"🤖 {name} — {desc}", expanded=False))

                # Run planning phase
                planning_results = crew.run_planning_phase(inputs)

                # Update statuses
                for container in status_containers:
                    container.update(state="complete")

                st.session_state.planning_results = planning_results
                st.session_state.current_phase = "plan_review"

                # Update DB
                update_project_status(project_id, "plan_ready", planning_results)

                st.session_state.page = "plan_review"
                st.rerun()

            except Exception as e:
                st.error(f"❌ Error during planning: {str(e)}")
                st.exception(e)


# =====================================================
# PAGE: PLAN REVIEW - Review Before Coding
# =====================================================
def render_plan_review():
    st.markdown("### 📋 Project Plan Review")
    st.markdown("*Review the AI-generated plan before starting development*")

    if not st.session_state.planning_results:
        st.warning("No planning results found. Please go back to Home.")
        if st.button("← Back to Home"):
            st.session_state.page = "home"
            st.rerun()
        return

    results = st.session_state.planning_results

    # Display planning results in tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📄 Requirements",
        "📋 PRD & Sitemap",
        "🎨 Design System",
        "✨ Animation Plan",
        "🏗️ Architecture",
    ])

    with tab1:
        st.markdown(results.get("requirements", "No data"))

    with tab2:
        st.markdown(results.get("prd", "No data"))

    with tab3:
        st.markdown(results.get("design_system", "No data"))

    with tab4:
        st.markdown(results.get("animation_plan", "No data"))

    with tab5:
        st.markdown(results.get("architecture", "No data"))

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("✅ Approve & Start AI Development", use_container_width=True, type="primary"):
            st.session_state.page = "developing"
            st.rerun()

    with col2:
        if st.button("🔄 Regenerate Plan", use_container_width=True):
            st.session_state.planning_results = None
            st.session_state.page = "home"
            st.rerun()


# =====================================================
# PAGE: DEVELOPING - Live Agent Tracking
# =====================================================
def render_developing():
    st.markdown("### 🚧 AI Development in Progress")
    st.markdown("*Your AI software house team is building your project...*")

    if not st.session_state.crew or not st.session_state.project_path:
        st.error("No active project. Please start from Home.")
        if st.button("← Back to Home"):
            st.session_state.page = "home"
            st.rerun()
        return

    # Agent activity panel
    st.markdown("### 🤖 Agent Activity")
    dev_agents = [
        ("Frontend Developer Agent", "Writing production-ready code..."),
        ("Code Reviewer Agent", "Reviewing code quality..."),
        ("QA Testing Agent", "Testing responsiveness & functionality..."),
        ("Self-Healing Debug Agent", "Fixing any issues found..."),
        ("Documentation Agent", "Generating README & docs..."),
    ]

    dev_statuses = []
    for name, desc in dev_agents:
        dev_statuses.append(st.status(f"🤖 {name} — {desc}", expanded=False))

    crew = st.session_state.crew
    inputs = {
        "user_idea": st.session_state.get("user_idea_input", ""),
        "project_path": st.session_state.project_path,
        "project_name": Path(st.session_state.project_path).name,
        "project_type": st.session_state.get("project_type_input", ""),
        "tech_preference": st.session_state.get("tech_preference_input", ""),
        "animation_level": st.session_state.get("animation_level_input", ""),
    }

    try:
        with st.spinner("🔨 Building your project... This may take a few minutes."):
            dev_results = crew.run_development_phase(inputs)

        # Mark all as complete
        for container in dev_statuses:
            container.update(state="complete")

        st.session_state.development_results = dev_results
        st.session_state.generation_complete = True

        # Update DB
        if st.session_state.project_id:
            update_project_status(st.session_state.project_id, "complete")

        st.success("🎉 Project generated successfully!")
        st.balloons()

        time.sleep(1)
        st.session_state.page = "workspace"
        st.rerun()

    except Exception as e:
        st.error(f"❌ Error during development: {str(e)}")
        st.exception(e)


# =====================================================
# PAGE: WORKSPACE - Project Viewer
# =====================================================
def render_workspace():
    if not st.session_state.project_path:
        st.warning("No project loaded. Please create or select a project.")
        if st.button("← Back to Home"):
            st.session_state.page = "home"
            st.rerun()
        return

    project_path = st.session_state.project_path
    project_name = Path(project_path).name

    st.markdown(f"### 🏗️ Project: `{project_name}`")

    # Top toolbar
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        try:
            zip_path = zip_project.run(project_path=project_path)
            if Path(zip_path).exists():
                with open(zip_path, "rb") as f:
                    st.download_button(
                        "📦 Download ZIP",
                        data=f.read(),
                        file_name=f"{project_name}.zip",
                        mime="application/zip",
                        use_container_width=True,
                    )
        except Exception:
            st.button("📦 ZIP (Error)", disabled=True, use_container_width=True)

    with col2:
        if st.button("👁️ Full Preview", use_container_width=True):
            st.session_state.page = "preview"
            st.rerun()

    with col3:
        if st.button("📋 View Plan", use_container_width=True):
            st.session_state.page = "plan_review"
            st.rerun()

    with col4:
        if st.button("🔄 Regenerate", use_container_width=True):
            st.session_state.page = "home"
            st.rerun()

    with col5:
        if st.button("🏠 Home", use_container_width=True):
            st.session_state.page = "home"
            st.rerun()

    st.markdown("---")

    # Three-column layout: File Tree | Code Viewer | Live Preview
    left_col, mid_col, right_col = st.columns([1, 2, 2])

    stats = get_project_stats(project_path)

    with left_col:
        st.markdown("#### 📁 File Explorer")

        if "error" not in stats:
            st.caption(f"{stats['total_files']} files · {stats['total_size_formatted']}")

            for file_info in stats["files"]:
                icon = get_file_icon(file_info["name"])
                size = format_file_size(file_info["size"])
                label = f"{icon} {file_info['relative_path']} ({size})"

                if st.button(
                    label,
                    key=f"file_{file_info['relative_path']}",
                    use_container_width=True,
                ):
                    st.session_state.selected_file = file_info["full_path"]
                    st.rerun()

            st.markdown("---")
            with st.expander("📂 Full File Tree"):
                tree = create_file_tree.run(project_path=project_path)
                st.code(tree, language="text")
        else:
            st.error("Could not load project files.")

    with mid_col:
        st.markdown("#### 💻 Code Viewer")

        if st.session_state.selected_file:
            file_path = st.session_state.selected_file
            file_name = Path(file_path).name
            language = get_file_language(file_name)

            st.caption(f"📄 {Path(file_path).relative_to(Path(project_path))}")

            content = read_file.run(file_path=file_path)
            if not content.startswith("Error:"):
                st.code(content, language=language, line_numbers=True)
            else:
                st.error(content)
        else:
            st.info("👈 Select a file from the explorer to view its code.")

    with right_col:
        st.markdown("#### 👁️ Live Preview")

        try:
            preview_html = build_preview_html.run(project_path=project_path)
            if not preview_html.startswith("<html><body><h1>Error"):
                import streamlit.components.v1 as components
                components.html(preview_html, height=600, scrolling=True)
            else:
                st.warning("Preview not available. No HTML file found.")
        except Exception as e:
            st.warning(f"Preview error: {str(e)}")


# =====================================================
# PAGE: FULL PREVIEW
# =====================================================
def render_preview():
    if not st.session_state.project_path:
        st.warning("No project loaded.")
        return

    col1, col2 = st.columns([1, 8])
    with col1:
        if st.button("← Back"):
            st.session_state.page = "workspace"
            st.rerun()
    with col2:
        st.markdown(f"### 👁️ Full Preview — `{Path(st.session_state.project_path).name}`")

    try:
        import streamlit.components.v1 as components
        preview_html = build_preview_html.run(project_path=st.session_state.project_path)
        components.html(preview_html, height=800, scrolling=True)
    except Exception as e:
        st.error(f"Preview error: {str(e)}")


# =====================================================
# PAGE: BUG FIXER
# =====================================================
def render_bug_fixer():
    st.markdown("### 🔧 AI Bug Fixer")
    st.markdown("*Upload your code files or paste code — our AI agents will find and fix issues.*")

    input_method = st.radio("Choose input method:", ["📁 Upload Files", "📝 Paste Code"], horizontal=True)

    uploaded_files = None
    code_input = None
    code_language = "html"

    if input_method == "📁 Upload Files":
        uploaded_files = st.file_uploader(
            "Upload your code files",
            type=["html", "htm", "css", "js", "py", "json", "txt"],
            accept_multiple_files=True,
        )

        if uploaded_files:
            st.markdown(f"**{len(uploaded_files)} file(s) uploaded:**")
            for f in uploaded_files:
                st.markdown(f"- {get_file_icon(f.name)} `{f.name}` ({format_file_size(f.size)})")
    else:
        code_input = st.text_area(
            "Paste your code here:",
            height=300,
            placeholder="Paste your HTML, CSS, JavaScript or Python code...",
        )
        code_language = st.selectbox("Language:", ["html", "css", "javascript", "python"])

    issue_description = st.text_area(
        "Describe the issue (optional):",
        placeholder="e.g., 'Navbar not responsive on mobile', 'Animation not triggering on scroll'...",
        height=80,
    )

    if st.button("🔍 Analyze & Fix with AI Agents", use_container_width=True, type="primary"):
        st.info("🤖 QA Agent and Debug Agent are analyzing your code...")

        from tools.code_tools import validate_html, validate_css, validate_javascript

        if input_method == "📁 Upload Files" and uploaded_files:
            for f in uploaded_files:
                content = f.read().decode("utf-8")
                st.markdown(f"#### Analysis of `{f.name}`:")

                if f.name.endswith((".html", ".htm")):
                    result = validate_html.run(html_code=content)
                    st.json(json.loads(result))
                elif f.name.endswith(".css"):
                    result = validate_css.run(css_code=content)
                    st.json(json.loads(result))
                elif f.name.endswith(".js"):
                    result = validate_javascript.run(js_code=content)
                    st.json(json.loads(result))

        elif input_method == "📝 Paste Code" and code_input:
            if code_language == "html":
                result = validate_html.run(html_code=code_input)
            elif code_language == "css":
                result = validate_css.run(css_code=code_input)
            elif code_language == "javascript":
                result = validate_javascript.run(js_code=code_input)
            else:
                result = json.dumps({"message": "Python validation coming soon"})

            st.json(json.loads(result))


# =====================================================
# PAGE: MY PROJECTS
# =====================================================
def render_projects():
    st.markdown("### 📁 My Projects")

    projects = get_all_projects()

    if not projects:
        st.info("No projects yet. Create your first project from the Home page!")
        if st.button("🏠 Go to Home"):
            st.session_state.page = "home"
            st.rerun()
        return

    for project in projects:
        status_icon = {
            "planning": "🔄",
            "plan_ready": "📋",
            "developing": "🚧",
            "complete": "✅",
        }.get(project["status"], "❓")

        with st.expander(f"{status_icon} {project['name']} — {project['status'].upper()}", expanded=False):
            col1, col2 = st.columns([3, 1])

            with col1:
                desc = project['description'] or "No description"
                st.markdown(f"**Description:** {desc[:200]}...")
                st.markdown(f"**Type:** {project['project_type']}")
                st.markdown(f"**Tech:** {project['tech_preference']}")
                st.markdown(f"**Animation:** {project['animation_level']}")
                st.markdown(f"**Created:** {project['created_at']}")
                st.markdown(f"**Path:** `{project['project_path']}`")

            with col2:
                if project["status"] == "complete" and project["project_path"]:
                    if st.button("📂 Open", key=f"open_{project['id']}"):
                        st.session_state.project_path = project["project_path"]
                        st.session_state.generation_complete = True
                        st.session_state.page = "workspace"
                        st.rerun()

                    if Path(project["project_path"]).exists():
                        try:
                            zip_path = zip_project.run(project_path=project["project_path"])
                            if Path(zip_path).exists():
                                with open(zip_path, "rb") as f:
                                    st.download_button(
                                        "📦 ZIP",
                                        data=f.read(),
                                        file_name=f"{project['name']}.zip",
                                        mime="application/zip",
                                        key=f"zip_{project['id']}",
                                    )
                        except Exception:
                            pass


# =====================================================
# PAGE: ABOUT
# =====================================================
def render_about():
    st.markdown("### ℹ️ About CodeForge AI")
    st.markdown("""
    **CodeForge AI** is an AI-powered virtual software house that takes your idea and 
    delivers a complete, working project — just like hiring an entire development team.
    
    ---
    
    #### 🤖 Our AI Agent Team
    
    | Agent | Role | What They Do |
    |---|---|---|
    | Business Analyst | Client Requirements | Understands your idea clearly |
    | Product Manager | Project Planning | Creates PRD, sitemap & MVP scope |
    | UI/UX Designer | Design System | Creates colors, fonts & components |
    | Motion Designer | Animations | Plans GSAP & scroll animations |
    | Frontend Architect | File Structure | Designs clean project architecture |
    | Frontend Developer | Code Generation | Writes production-ready code |
    | Code Reviewer | Quality Assurance | Ensures code quality & best practices |
    | QA Engineer | Testing | Tests responsiveness & functionality |
    | Debug Engineer | Bug Fixing | Auto-fixes any issues found |
    | Technical Writer | Documentation | Generates README & deployment guide |
    
    ---
    
    #### 🛠️ Built With
    
    - **CrewAI** — Multi-agent orchestration framework
    - **Groq** — Ultra-fast LLM inference (Llama 3.3 70B)
    - **Streamlit** — Beautiful web UI
    - **Python** — Backend logic & tooling
    
    ---
    
    #### 🌟 What Makes Us Different
    
    Unlike Cursor, Bolt, or v0 which generate code from a single prompt, CodeForge AI:
    
    1. **Plans first, codes later** — just like a real agency
    2. **Multiple specialized agents** — each with their own expertise
    3. **Self-healing code** — agents review, test & fix their own output
    4. **Complete deliverables** — code + docs + preview + ZIP download
    
    ---
    
    *Built with ❤️ using AI Agent Technology*
    """)


# =====================================================
# MAIN ROUTER
# =====================================================
def main():
    page = st.session_state.page

    if page == "home":
        render_home()
    elif page == "plan_review":
        render_plan_review()
    elif page == "developing":
        render_developing()
    elif page == "workspace":
        render_workspace()
    elif page == "preview":
        render_preview()
    elif page == "bug_fixer":
        render_bug_fixer()
    elif page == "projects":
        render_projects()
    elif page == "about":
        render_about()
    else:
        render_home()


if __name__ == "__main__":
    main()