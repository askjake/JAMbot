from typing import List

SUGGESTIONS = {
    "git_clone": ["Run the test suite", "Check dependencies", "Scan for security issues", "Show project structure"],
    "python_run": ["Show created files", "Run with different params", "Profile performance"],
    "kubectl_get": ["Check pod logs", "Describe failing pods", "Check resource usage"],
    "aws_s3": ["List bucket contents", "Check bucket policy", "Download a file"],
    "file_created": ["Preview file contents", "Download the file", "Run analysis on it"],
    "error_occurred": ["Show detailed logs", "Try alternative approach", "Break into smaller steps"],
}


def get_suggestions(tool_name: str, tool_output: str) -> List[str]:
    key = None
    if "clone" in tool_name: key = "git_clone"
    elif "run_python" in tool_name: key = "python_run"
    elif "kubectl" in tool_output.lower(): key = "kubectl_get"
    elif "s3" in tool_output.lower(): key = "aws_s3"
    elif any(x in tool_output.lower() for x in ["created", "wrote", "saved"]): key = "file_created"
    elif any(x in tool_output.lower() for x in ["error", "failed", "exception"]): key = "error_occurred"
    return SUGGESTIONS.get(key, [])[:3]


def format_suggestions(suggestions: List[str]) -> str:
    if not suggestions: return ""
    header = chr(10) + chr(10) + "---" + chr(10) + "**What would you like to do next?**"
    items = chr(10).join(f"{i+1}. {s}" for i, s in enumerate(suggestions))
    return header + chr(10) + items