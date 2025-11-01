# Crowd_Source_Waste_Management
A data-driven web application built using Streamlit to facilitate crowd-sourced reporting of waste issues, leveraging simulated AI for automated prioritization and routing to municipal teams.

🌟 Features

This application implements a complete reporting and dispatch workflow based on the project’s core flowchart:

📝 Report Issue – Users submit location, waste type, and description via a simple web form.

🤖 Predict Problem Areas & Validate Report – The system checks the report’s validity (e.g., sufficient detail in location/description) and matches it with an internal dataset (problem_areas.csv) to categorize the report’s area.

🚚 AI Routing and Prioritization – Based on waste type and location, the system assigns a Priority (Critical, High, Medium) and routes it to the appropriate Dispatch Team (Hazardous Response, Local Crew, etc.).

📊 View Reports Dashboard – An administrative dashboard displays all submitted reports, their current status, and allows simulated resolution by marking issues as Resolved.

⚙️ Installation and Setup
🧩 Prerequisites

Python 3.8 or higher installed on your system.

🪜 Steps
1. Clone the Repository
git clone https://github.com/YourUsername/YourRepoName.git
cd YourRepoName

2. Install Dependencies
pip install streamlit pandas

3. Create the Dataset

The application requires a file named problem_areas.csv in the root directory.
This dataset simulates the AI’s prediction and routing logic.

Example Structure:

Location_Keyword	Waste_Type	Priority_Level	Dispatch_Team
Park	General Trash	Medium	Local Crew
Highway	Construction Debris	High	Special Haulage
Riverbank	Hazardous	Critical	Hazardous Response
🚀 Running the Application

Once setup is complete, run:

streamlit run app.py


Streamlit will launch the app in your default browser, usually at:
👉 http://localhost:8501

💾 Data Persistence

All submitted reports are automatically saved in waste_reports.csv in the same directory.
This ensures that your data remains available between application sessions.

🧱 Project Structure
File	Purpose
app.py	Contains the entire Streamlit app logic, UI layout, and simulation functions (simulate_validation, simulate_routing_and_prioritization).
problem_areas.csv	Input dataset used by the AI simulation for lookup and prioritization rules.
waste_reports.csv	Output file storing all submitted reports.
README.md	Project documentation file (this one).
🤝 Contributing

Contributions are welcome!
If you’d like to enhance the validation logic, improve the AI simulation, or upgrade the Streamlit UI, feel free to:

🐛 Open an issue

💡 Submit a pull request
