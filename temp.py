import mimetypes
import contextlib
from time import time
import tempfile
import traceback
import contextlib
import modal
import io
import traceback
import os
import tempfile
import time
import mimetypes

app = modal.App("analystGPT")

image = (
    modal.Image.debian_slim()
    .pip_install(
        "pandas",
        "numpy",
        "matplotlib",
        "scikit-learn"
    )
)

code = """
import os
import matplotlib.pyplot as plt

os.makedirs("plots", exist_ok=True)

plt.plot([1,2,3])
plt.savefig("plots/image.png")
"""

@app.function(
    image=image, 
    timeout=30,
    memory=1024,
    cpu=1 
)
def execute_python(code: str, files: dict[str, bytes]):
    start = time.time()
    stdout = io.StringIO()
    stderr = io.StringIO()
    
    cwd = os.getcwd()

    with tempfile.TemporaryDirectory() as tempdir:
        os.chdir(tempdir)
        for filename, content in files.items():
            with open(filename, "wb") as f:
                f.write(content)
    
    
        try:
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                exec(code, {})
            success = True
            error = ""

        except Exception:
            success = False
            error = traceback.format_exc()

        finally:
            os.chdir(cwd)

        returned_files = {}

        for root, _, filenames in os.walk(tempdir):
            for filename in filenames:
                filepath = os.path.join(root, filename)

                relative_path = os.path.relpath(filepath, tempdir)

                if relative_path in files:
                    continue
                    
                with open(filepath, "rb") as f:
                    data = f.read()

                    mime, _ = mimetypes.guess_type(relative_path)

                    returned_files[relative_path] = {
                        "content": data,
                        "size": len(data),
                        "extension": os.path.splitext(relative_path)[1],
                        "mime": mime
                    }

    
    execution_time = time.time() - start
    
    return {
        "status": success,
        "stdout": stdout.getvalue(),
        "stderr": stderr.getvalue() + error,
        "execution_time": execution_time,
        "files": returned_files,
    }

@app.local_entrypoint()
def main():
    with open("people.csv", "rb") as f:
        file_bytes = f.read()
    
    try:
        result = execute_python.remote(
            code,
            {
                "people.csv": file_bytes
            }
        )

        if not result["status"]:
            print("Exception Failed:")
            print(result["stderr"])
            return
            
        for filename, info in result["files"].items():
            os.makedirs(os.path.dirname(filename), exist_ok=True) if os.path.dirname(filename) else None
            with open(filename, "wb") as f:
                f.write(info["content"])

        print("Files saved locally!")
        print(result)

    except Exception:
        print(traceback.format_exc())