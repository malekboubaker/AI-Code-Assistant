import os
import sys

# Ensure backend imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.agent.orchestrator import AgentOrchestrator
from backend.api.schemas import GenerateRequest

def main():
    orchestrator = AgentOrchestrator()
    request = GenerateRequest(
        instruction="Refactor the configuration constants from settings.py and put them in a new file constants.py. Both files need to be modified.",
        code="""
PORT = 8000
HOST = "127.0.0.1"

def start_server():
    print(f"Starting at {HOST}:{PORT}")
""",
        language="python",
        file_path="settings.py",
        project_path=os.getcwd(),
        use_rag=False
    )
    
    response = orchestrator.run(request)
    print("Task Assigned:", response.task)
    if response.edits:
        print(f"Found {len(response.edits)} edits:")
        for edit in response.edits:
            print(f"File: {edit.file_path}")
            print(f"Reason: {edit.reason}")
            print(f"Content length: {len(edit.new_content)}")
            print("---")
    else:
        print("No multi-file edits generated.")
        print("Raw explanation:")
        print(response.explanation)
        print("Raw code:")
        print(response.generated_code)

if __name__ == "__main__":
    main()

