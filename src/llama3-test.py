import json
from llamaapi import LlamaAPI

# Initialize the SDK
llama = LlamaAPI(
    api_token="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6IjlmZTgyZmU0LWExY2ItNDEwYy1iMTk0LTE5NDVjZWFhMzQyMSJ9.0KtCHeJs8vm4-AwMeldOdG4E4H6P8FvqSvH6SBvUlCc",
    hostname="http://localhost:11434",
    domain_path="/api/chat"
    )

# Build the API request
# api_request_json = {
#     "model": "llama3:8b",
#     "messages": [
#         {"role": "user", "content": "What is the weather like in Boston?"},
#     ],
#     "functions": [
#         {
#             "name": "get_current_weather",
#             "description": "Get the current weather in a given location",
#             "parameters": {
#                 "type": "object",
#                 "properties": {
#                     "location": {
#                         "type": "string",
#                         "description": "The city and state, e.g. San Francisco, CA",
#                     },
#                     "days": {
#                         "type": "number",
#                         "description": "for how many days ahead you wants the forecast",
#                     },
#                     "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
#                 },
#             },
#             "required": ["location", "days"],
#         }
#     ],
#     "stream": False,
#     "function_call": "get_current_weather",
# }

api_request_json = {
    "model": "llama3:8b",
    "messages": [
        {"role": "user", "content": "llama是否支持调用函数查询外部数据"},
    ],
    "stream": False,
}

# Execute the Request
response = llama.run(api_request_json)
print(json.dumps(response.json(), indent=2))
print(response.json().get("message").get("content"))

