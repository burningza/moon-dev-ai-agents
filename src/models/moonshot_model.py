"""
🌙 Moon Dev's Moonshot Model Integration
Built with love by Moon Dev 🚀

This module provides integration with Moonshot AI's API.
Moonshot is a powerful reasoning model optimized for complex analysis.

Docs: https://platform.moonshot.ai/docs/overview
"""

from openai import OpenAI
from termcolor import cprint
from .base_model import BaseModel, ModelResponse

class MoonshotModel(BaseModel):
    """Implementation for Moonshot API models"""
    
    # Available Moonshot models 
    # From here https://platform.moonshot.ai/docs/guide/kimi-k2-quickstart#recommended-api-versions
    AVAILABLE_MODELS = [
        "moonshot-v1-8k",
        "moonshot-v1-32k",
        "moonshot-v1-128k",
        "kimi-k2-thinking",
        "kimi-k2-thinking-turbo",  # Recommended for complex reasoning
        "kimi-k2-0905-preview",
        "kimi-k2-turbo-preview"
    ]
    
    def __init__(self, api_key=None, model_name="kimi-k2-thinking"):
        """Initialize Moonshot model
        
        Args:
            api_key: Moonshot API key (if not provided, loads from env)
            model_name: Name of the Moonshot model to use
        """
        self.base_url = "https://api.moonshot.ai/v1"
        self.model_name = model_name
        super().__init__(api_key=api_key)
        self.initialize_client()
    
    def initialize_client(self):
        """Initialize the Moonshot client connection"""
        if not self.api_key:
            cprint("❌ Moonshot API key not found!", "white", "on_red")
            cprint("   Please set MOONSHOT_API_KEY in your .env file", "yellow")
            raise ValueError("MOONSHOT_API_KEY not configured")
        
        try:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
            )
            # Test the connection by listing models
            self.client.models.list()
            cprint(f"✨ Successfully connected to Moonshot API", "green")
            cprint(f"   Model: {self.model_name}", "cyan")
        except Exception as e:
            cprint(f"❌ Error initializing Moonshot: {str(e)}", "white", "on_red")
            raise
    
    @property
    def model_type(self):
        """Return the model type identifier"""
        return "moonshot"
    
    def is_available(self):
        """Check if the model is available"""
        try:
            self.client.models.list()
            return True
        except:
            return False
    
    def generate_response(self, system_prompt, user_content, temperature=0.7, max_tokens=None, **kwargs):
        """Generate a response using Moonshot API
        
        Args:
            system_prompt: System prompt/instructions
            user_content: User's query/content
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters
            
        Returns:
            ModelResponse object with the generated content
        """
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )

            content = completion.choices[0].message.content
            
            # Return as ModelResponse
            return ModelResponse(
                content=content,
                raw_response=completion.model_dump(),
                model_name=self.model_name,
                usage={"total_tokens": completion.usage.total_tokens}
            )
            
        except Exception as e:
            raise Exception(f"Moonshot API error: {str(e)}")
    
    def __str__(self):
        """String representation"""
        return f"MoonshotModel({self.model_name})"
    
    def get_model_parameters(self, model_name=None):
        """Get parameters for a specific Moonshot model"""
        model = model_name or self.model_name
        
        params = {
            "moonshot-v1-8k": {
                "context_window": 8000,
                "max_output_tokens": 4096,
            },
            "moonshot-v1-32k": {
                "context_window": 32000,
                "max_output_tokens": 8000,
            },
            "moonshot-v1-128k": {
                "context_window": 128000,
                "max_output_tokens": 8000,
            },
            "kimi-k2-thinking": {
                "context_window": 256000,
                "max_output_tokens": 8000,
            },
        }
        
        return params.get(model, {})