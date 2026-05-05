# Smart Digital Twin with AI-Enhanced Defect Detection for Aerospace Manufacturing

BSc. (Hons.) Computer Science Final Year Project 
Birmingham City University

Author: Nissy Joseph (22177041)
Supervisor: Debashish Das
Industry Partners: Safran - Rhys Woodward

## What This Project Does
A prototype digital twin of a 5-axis CNC machining process for Ti-6Al-4V
aerospace components, with AI-based defect detection using Random Forest
(parameter prediction) and YOLOv11 OBB (visual inspection).

## Setup Instructions

### 1. Install Python

Download Python 3.12 from https://www.python.org/downloads/
During installation, tick "Add python.exe to PATH"

### 2. Clone or Download This Project

git clone https://github.com/nissyjoseph/digital-twin-project.git
cd digital-twin-project

Or download the ZIP and extract it.

### 3. Create Virtual Environment
Windows:
python -m venv venv
venv\Scripts\activate

Mac/Linux:
python3 -m venv venv
source venv/bin/activate

### 4. Install Dependencies
pip install -r requirements.txt

### 6. Run the Digital Twin Simulation
python simulation.py
Output files are saved to the `data/` directory.

### 7. Run the Dashboard
streamlit run dashboard.py
Opens in browser at http://localhost:8501


### 8. Detailed video of the working prototype has been uploaded along with the files

## Project Structure
digital-twin-project/
├── config.py             # Machine parameters, sensor thresholds, ISO constants
├── sensors.py            # Physics-based sensor generation with Cholesky correlations
├── simulation.py         # SimPy discrete-event simulation engine
├── supabase_client.py    # Database integration layer
├── dashboard.py          # Streamlit visualisation dashboard
├── requirements.txt      # Python package dependencies
├── .env                  # Supabase credentials 
├── .gitignore            # Files excluded from version control
├── data/                 # Simulation output CSVs (generated)
└── venv/                 # Virtual environment (generated)


## Technologies

- Python 3.12 (Python 3.12.10 (tags/v3.12.10:0cc8128, Apr  8 2025, 12:21:36) [MSC v.1943 64 bit (AMD64)] on win32), 
- SimPy, NumPy, SciPy, Pandas
- Supabase (PostgreSQL), Streamlit, Plotly
- scikit-learn (Random Forest), Ultralytics YOLOv11 OBB
- ISO 23247, ISO 10816-3, ISO 3685
