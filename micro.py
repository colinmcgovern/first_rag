import boto3
import json

prompt_data = """
say hello in spanish
"""

bedrock=boto3.client(service_name='bedrock-runtime')

payload={
        "messages": [
            {
                "role": "user",
                "content": [{"text": "say hello in spanish"}]
            }
        ]
    }
body=json.dumps(payload)
model_id="us.amazon.nova-micro-v1:0"
response=bedrock.invoke_model_with_response_stream(
    body=body,
    modelId=model_id,
    # accept="application/json",
    # contentType="application/json"
)

# response_body=json.loads(next(response.get("body")).read())
# response_text=response_body['generation']
# print(response_text)
for event in response['body']:
    chunk = event.get("chunk")
    if chunk:
        
        chunks = chunk["bytes"].decode("utf-8")
        try:
            parsed_chunk = json.loads(chunks)
            if "contentBlockDelta" in parsed_chunk and "delta" in parsed_chunk["contentBlockDelta"] and "text" in parsed_chunk["contentBlockDelta"]["delta"]:
                chunks = parsed_chunk["contentBlockDelta"]["delta"]["text"]
                print(chunks, end='', flush=True)
        except json.JSONDecodeError:
            pass  # Keep original chunks if it's not valid JSON