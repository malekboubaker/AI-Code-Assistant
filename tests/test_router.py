import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from backend.agent.task_router import TaskRouterAgent

router = TaskRouterAgent()
print("Task:", router.detect("Generate a simple AI Travel Planner web application from scratch. Create an index.html with a modern UI layout for entering a destination and dates, a styles.css file to make it look beautiful with glassmorphism effects, and an app.js file with a mock function that simulates fetching an itinerary. Output all three files at once."))

