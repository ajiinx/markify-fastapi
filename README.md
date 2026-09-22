# Autoasses
> In development

# Current Implementations
- OCR Pipeline (Student Answer Sheet -> Extracted Text)
- LLM Based Segmentation(Fallback: regex expression segmentation)
- Model answer conversion and storage(markdown format)
- Database support(MongoDB)
- Smart per-student answersheet checking using model answer as ground truth
- UI support in React.js for API enpoints

# API endpoints

![API Enpoints explained](./assets/Markify_API_Endpoints.png)


# Directory structure

```bash
.
├── assets
│   └── Markify_API_Endpoints.png
├── backend
│   ├── db.py
│   ├── llm_grading.py
│   ├── llm_marking_scheme.py
│   ├── llm_segmentation.py
│   ├── main.py
│   ├── model_client.py
│   ├── model_server.py
│   ├── olmocr_grading
│   │   ├── config.py
│   │   ├── image_utils.py
│   │   ├── __init__.py
│   │   ├── ocr_engine.py
│   │   ├── output_writer.py
│   │   ├── pdf_utils.py
│   │   └── prompts.py
│   ├── requirements.txt
│   └── segmentation.py
├── frontend
│   ├── bun.lock
│   ├── components.json
│   ├── index.html
│   ├── jsconfig.json
│   ├── package.json
│   ├── package-lock.json
│   ├── public
│   │   ├── favicon.svg
│   │   └── icons.svg
│   ├── README.md
│   ├── src
│   │   ├── App.jsx
│   │   ├── assets
│   │   │   ├── hero.png
│   │   │   ├── react.svg
│   │   │   └── vite.svg
│   │   ├── components
│   │   │   ├── common
│   │   │   │   └── FileUploadZone.jsx
│   │   │   ├── dashboard
│   │   │   │   └── DashboardPage.jsx
│   │   │   ├── layout
│   │   │   │   └── AppLayout.jsx
│   │   │   ├── model-answer
│   │   │   │   └── ModelAnswerPage.jsx
│   │   │   ├── student-eval
│   │   │   │   └── StudentEvaluationPage.jsx
│   │   │   ├── system-status
│   │   │   │   └── SystemStatusPage.jsx
│   │   │   └── ui
│   │   │       ├── accordion.jsx
│   │   │       ├── alert-dialog.jsx
│   │   │       ├── alert.jsx
│   │   │       ├── aspect-ratio.jsx
│   │   │       ├── attachment.jsx
│   │   │       ├── avatar.jsx
│   │   │       ├── badge.jsx
│   │   │       ├── breadcrumb.jsx
│   │   │       ├── bubble.jsx
│   │   │       ├── button-group.jsx
│   │   │       ├── button.jsx
│   │   │       ├── calendar.jsx
│   │   │       ├── card.jsx
│   │   │       ├── carousel.jsx
│   │   │       ├── chart.jsx
│   │   │       ├── checkbox.jsx
│   │   │       ├── collapsible.jsx
│   │   │       ├── combobox.jsx
│   │   │       ├── command.jsx
│   │   │       ├── context-menu.jsx
│   │   │       ├── dialog.jsx
│   │   │       ├── direction.jsx
│   │   │       ├── drawer.jsx
│   │   │       ├── dropdown-menu.jsx
│   │   │       ├── empty.jsx
│   │   │       ├── field.jsx
│   │   │       ├── hover-card.jsx
│   │   │       ├── input-group.jsx
│   │   │       ├── input.jsx
│   │   │       ├── input-otp.jsx
│   │   │       ├── item.jsx
│   │   │       ├── kbd.jsx
│   │   │       ├── label.jsx
│   │   │       ├── marker.jsx
│   │   │       ├── menubar.jsx
│   │   │       ├── message.jsx
│   │   │       ├── message-scroller.jsx
│   │   │       ├── native-select.jsx
│   │   │       ├── navigation-menu.jsx
│   │   │       ├── pagination.jsx
│   │   │       ├── popover.jsx
│   │   │       ├── progress.jsx
│   │   │       ├── questionnaire.jsx
│   │   │       ├── radio-group.jsx
│   │   │       ├── resizable.jsx
│   │   │       ├── scroll-area.jsx
│   │   │       ├── select.jsx
│   │   │       ├── separator.jsx
│   │   │       ├── sheet.jsx
│   │   │       ├── sidebar.jsx
│   │   │       ├── skeleton.jsx
│   │   │       ├── slider.jsx
│   │   │       ├── spinner.jsx
│   │   │       ├── switch.jsx
│   │   │       ├── table.jsx
│   │   │       ├── tabs.jsx
│   │   │       ├── textarea.jsx
│   │   │       ├── toast.jsx
│   │   │       ├── toggle-group.jsx
│   │   │       ├── toggle.jsx
│   │   │       └── tooltip.jsx
│   │   ├── context
│   │   │   └── AssessmentContext.jsx
│   │   ├── hooks
│   │   │   └── use-mobile.js
│   │   ├── index.css
│   │   ├── lib
│   │   │   └── utils.js
│   │   ├── main.jsx
│   │   └── services
│   │       └── api.js
│   └── vite.config.js
└── README.md

20 directories, 105 files
```

# Setup

1. Frontend
```bash
# Using npm
cd frontend && npm install 
npm run dev

# Using bun
cd frontend && bun install
bun run dev
```

2. Backend
```bash
# Windows
cd backend
uv venv

# Activating the enviroment
.venv\Scripts\activate

#Installing the dependancies
uv pip install -r .\requirements.txt

------
# Ubuntu
cd backend
uv venv

# Activating the enviroment
source venv/bin/activate

# Installing the dependancies
uv pip install -r requirements.txt
```

3. Running the backend
```bash
# Running the model server (vLLM)
vllm serve Qwen/Qwen2.5-VL-7B-Instruct --host 127.0.0.1 --port 8001 --gpu-memory-utilization 0.85

# Running the main server
uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```