import os
from swarms import Agent
from swarm_models import OpenAIChat
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get the OpenAI API key from the environment variable
api_key = os.getenv("OPENAI_API_KEY")

# Define the agent for crowd metrics
crowd_metrics_agent = Agent(
    agent_name="Crowd-Metrics-Agent",
    system_prompt="Manage crowd metrics analysis.",
    llm=OpenAIChat(openai_api_key=api_key, model_name="gpt-4o-mini", temperature=0.1),
    max_loops=1,
    autosave=True,
    dashboard=False,
    verbose=True,
    dynamic_temperature_enabled=True,
    saved_state_path="crowd_metrics_agent.json",
    user_name="swarms_corp",
    retry_attempts=1,
    context_length=200000,
    return_step_meta=False,
    output_type="string",
    streaming_on=False,
)

# Define the agent for the DJ system
dj_system_agent = Agent(
    agent_name="DJ-System-Agent",
    system_prompt="Control the DJ system.",
    llm=OpenAIChat(openai_api_key=api_key, model_name="gpt-4o-mini", temperature=0.1),
    max_loops=1,
    autosave=True,
    dashboard=False,
    verbose=True,
    dynamic_temperature_enabled=True,
    saved_state_path="dj_system_agent.json",
    user_name="swarms_corp",
    retry_attempts=1,
    context_length=200000,
    return_step_meta=False,
    output_type="string",
    streaming_on=False,
)

def main():
    """Main function to orchestrate the agents."""
    print("Starting Crowd Metrics Agent...")
    crowd_metrics_agent.run("Initialize crowd metrics analysis.")

    print("Starting DJ System...")
    dj_system_agent.run("Initialize DJ system.")

    try:
        # Keep the main program running while both agents are active
        while True:
            time.sleep(1)  # Sleep to reduce CPU usage
    except KeyboardInterrupt:
        print("Shutting down...")
        crowd_metrics_agent.terminate()
        dj_system_agent.terminate()
        print("Agents terminated.")

if __name__ == "__main__":
    main()
