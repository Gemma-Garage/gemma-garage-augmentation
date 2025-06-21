from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from endpoints import data_parser, augmentation

app = FastAPI(title="LLM Garage Data Augmentation API")

origins = [
    "http://localhost:3000",  # your React app's origin
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(data_parser.router, prefix="/parsing", tags=["Document Parsing"])
app.include_router(augmentation.router, prefix="/augment", tags=["Data Augmentation"])

def update_api_key_in_config_file(config_path: str, new_api_key: str):
    """
    Safely updates the api_key under the 'api-endpoint' section in a YAML file,
    preserving comments and formatting by treating it as a text file.

    Args:
        config_path: The path to the config.yaml file.
        new_api_key: The new API key to set.
    """
    try:
        with open(config_path, 'r') as f:
            lines = f.readlines()

        new_lines = []
        in_api_endpoint_section = False
        key_updated = False

        for line in lines:
            # Determine if we are in the api-endpoint section
            if line.strip().startswith('api-endpoint:'):
                in_api_endpoint_section = True
            # Check if we are leaving the section (by finding another non-indented key)
            elif in_api_endpoint_section and not line.startswith(' ') and not line.startswith('#') and line.strip():
                in_api_endpoint_section = False

            # If in the right section and the line contains api_key, replace its value
            if not key_updated and in_api_endpoint_section and 'api_key:' in line:
                # This regex replaces the value between the quotes after 'api_key:'
                pattern = re.compile(r'(api_key:\s*[""]).*?([""])')
                new_line = pattern.sub(f'\\1{new_api_key}\\2', line)
                new_lines.append(new_line)
                key_updated = True
            else:
                new_lines.append(line)

        if key_updated:
            with open(config_path, 'w') as f:
                f.writelines(new_lines)
            print(f"Successfully updated API key in {config_path}")
        else:
            print(f"Warning: 'api_key' under 'api-endpoint' section not found. File not changed.")

    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}")
    except Exception as e:
        print(f"An error occurred while updating the config file: {e}")

if __name__ == "__main__":
    import uvicorn
    import os
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    print(f"Gemini API key: {GEMINI_API_KEY}")
    update_api_key_in_config_file("synthetic-data-kit/synthetic_data_kit/config.yaml", GEMINI_API_KEY)
    uvicorn.run(app, host="0.0.0.0", port=8000)
