# importing other functionality -->
from utils.model_loader import ModelLoader
from prompt_library.prompt import SYSTEM_PROMPT
from langgraph.graph import StateGraph, MessagesState,END, START
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import AIMessage

# Importing tools for helping Agent in Functioning ---> 
from tools.weather_info_tool import WeatherInfoTool
from tools.place_search_tool import PlaceSearchTool
from tools.expense_calculator_tool import CalculatorTool
from tools.currency_conversion_tool import CurrencyConverterTool 


class GraphBuilder():

    def __init__(self, model_choice: str = "groq"):
        self.model_choice = model_choice
        self.model_loader = ModelLoader(model_choice=model_choice)
        self.llm = self.model_loader.load_llm()
        self.tools = []
        
        self.weather_tools = WeatherInfoTool()
        self.place_search_tools = PlaceSearchTool()
        self.calculator_tools = CalculatorTool()
        self.currency_converter_tool = CurrencyConverterTool()

        # Collect all tool callables from each tool wrapper
        self.tools.extend([
            *self.weather_tools.weather_tool_list,
            *self.place_search_tools.place_search_tool_list,
            *self.calculator_tools.calculator_tool_list,
            *self.currency_converter_tool.currency_converter_tool_list,
        ])
        

        # Models that do NOT support tool calling — use raw LLM invocation for these
        _NO_TOOL_MODELS = {"compound-beta", "compound-beta-mini"}
        if model_choice.split("/")[-1] in _NO_TOOL_MODELS:
            # compound-beta doesn't accept bind_tools; call it directly
            self.llm_with_tools = self.llm
        else:
            self.llm_with_tools = self.llm.bind_tools(tools=self.tools)
        
        self.graph = None
        
        self.system_prompt = SYSTEM_PROMPT


    def agent_function(self, state):

            """Main agent function: forward the incoming messages to the LLM (with tools)
            and return the LLM's response so the caller can extract the generated plan.
            """
            try:
                user_messages = None
                if isinstance(state, dict):
                    user_messages = state.get("messages") or state.get("messages", [])
                elif hasattr(state, "messages"):
                    user_messages = getattr(state, "messages")
                if user_messages is None:
                    user_messages = []

                input_question = [self.system_prompt] + list(user_messages)
                result = self.llm_with_tools.invoke(input_question)

                # Extract textual content from common result shapes
                text = None
                if isinstance(result, dict):
                    if "messages" in result and result["messages"]:
                        last = result["messages"][-1]
                        text = getattr(last, "content", None) or str(last)
                    elif "content" in result:
                        text = result["content"]
                    else:
                        text = str(result)
                elif hasattr(result, "content"):
                    text = result.content
                else:
                    text = str(result)

                # Return a dict with `messages` containing an AIMessage instance so downstream
                # ToolNode finds an AIMessage as the last message. Preserve any tool_calls
                ai_msg = AIMessage(content=text)
                # If the raw result included tool_calls, attach them to the AIMessage
                try:
                    setattr(ai_msg, "tool_calls", getattr(result, "tool_calls", []))
                except Exception:
                    pass
                return {"messages": [ai_msg]}
            except Exception as e:
                ai_msg = AIMessage(content=f"Error: {e}")
                try:
                    setattr(ai_msg, "tool_calls", [])
                except Exception:
                    pass
                return {"messages": [ai_msg]}


    def build_graph(self):
        graph_builder = StateGraph(MessagesState)
        graph_builder.add_node("agent", self.agent_function)
        graph_builder.add_node("tools", ToolNode(self.tools))
        graph_builder.add_edge(START, "agent")
        # Route to tools only when the last AI message contains tool_calls.
        # This prevents an unconditional cycle between `agent` and `tools`.
        graph_builder.add_conditional_edges("agent", tools_condition)
        graph_builder.add_edge("tools", "agent")
        graph_builder.add_edge("agent", END)

        self.graph = graph_builder.compile()
        return self.graph
    

    def __call__(self):
        return self.build_graph()



