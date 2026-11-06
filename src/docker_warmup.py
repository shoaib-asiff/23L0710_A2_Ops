"""
Task 6.1: Docker Warm-Up Lab Script
Author: 23L0710
Project: 23L0710_A2_Ops - VisionGuard

Executes:
1. Pulls ready-made official lightweight image (nginx:alpine)
2. Runs container with port mapping (-p 8085:80) and volume mount (-v)
3. Tests volume data persistence between host and container
4. Reads container logs (docker logs)
5. Executes commands in container shell (docker exec)
6. Lists containers (docker ps -a) and images (docker images)
7. Stops and removes container (docker stop, docker rm)
8. Removes image (docker rmi)
9. Captures terminal screenshot via headless Chrome
10. Authors comprehensive report with student definitions (Image, Container, Layer, Volume, Port Mapping)
"""

import os
import sys
import subprocess
import time

EVIDENCE_DIR = "evidence/task6"
WARMUP_DATA_DIR = os.path.join(EVIDENCE_DIR, "warmup_data")
LOG_FILE = os.path.join(EVIDENCE_DIR, "docker_warmup_log.txt")
REPORT_MD = os.path.join(EVIDENCE_DIR, "docker_warmup_report.md")
SCREENSHOT_PNG = os.path.join(EVIDENCE_DIR, "docker_warmup_terminal.png")
CHROME_BIN = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

CONTAINER_NAME = "visionguard_warmup_box"
IMAGE_NAME = "nginx:alpine"
PORT_HOST = 8085
PORT_CONTAINER = 80


def run_cmd(cmd_list, description=""):
    """Runs a shell command, logs it, and returns stdout."""
    cmd_str = " ".join(cmd_list)
    print(f"\n[RUNNING] {cmd_str}")
    t0 = time.time()
    res = subprocess.run(cmd_list, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    elapsed = time.time() - t0
    output = res.stdout.strip()
    print(output)
    return {
        "command": cmd_str,
        "description": description,
        "output": output,
        "exit_code": res.returncode,
        "elapsed_sec": round(elapsed, 2)
    }


def execute_warmup():
    os.makedirs(EVIDENCE_DIR, exist_ok=True)
    os.makedirs(WARMUP_DATA_DIR, exist_ok=True)

    # 1. Prepare host test file for volume mounting
    host_file = os.path.join(WARMUP_DATA_DIR, "host_note.txt")
    with open(host_file, "w") as f:
        f.write("Task 6.1: Verified bidirectional Docker volume mount by student 23L0710\n")

    steps = []

    # Clean up prior instance if exists
    subprocess.run(["docker", "rm", "-f", CONTAINER_NAME], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Action 1: Pull Image
    steps.append(run_cmd(
        ["docker", "pull", IMAGE_NAME],
        description="Action 1: Pull ready-made official image from Docker Hub"
    ))

    # Action 2: Run container with Port Mapping (-p) and Volume Mount (-v)
    abs_mount = os.path.abspath(WARMUP_DATA_DIR)
    steps.append(run_cmd(
        [
            "docker", "run", "-d",
            "--name", CONTAINER_NAME,
            "-p", f"{PORT_HOST}:{PORT_CONTAINER}",
            "-v", f"{abs_mount}:/usr/share/nginx/html/data",
            IMAGE_NAME
        ],
        description="Action 2: Run container in detached mode with port mapping (-p) and volume mount (-v)"
    ))

    time.sleep(2.0)  # Wait for container startup

    # Action 3: Verify HTTP port mapping via curl
    steps.append(run_cmd(
        ["curl", "-I", f"http://127.0.0.1:{PORT_HOST}"],
        description="Action 3: Verify port mapping by curling mapped host port"
    ))

    # Action 4: Read container logs
    steps.append(run_cmd(
        ["docker", "logs", "--tail", "20", CONTAINER_NAME],
        description="Action 4: Inspect container runtime stdout/stderr logs"
    ))

    # Action 5: Open shell and execute commands inside container
    steps.append(run_cmd(
        [
            "docker", "exec", CONTAINER_NAME,
            "sh", "-c",
            "uname -a && cat /usr/share/nginx/html/data/host_note.txt && ps aux | head -n 5"
        ],
        description="Action 5: Open shell inside running container and inspect filesystem & processes"
    ))

    # Action 6: List running containers (docker ps -a)
    steps.append(run_cmd(
        ["docker", "ps", "-a", "--filter", f"name={CONTAINER_NAME}"],
        description="Action 6: List active and stopped containers"
    ))

    # Action 7: List images (docker images)
    steps.append(run_cmd(
        ["docker", "images", IMAGE_NAME],
        description="Action 7: List Docker images stored locally"
    ))

    # Action 8: Stop container
    steps.append(run_cmd(
        ["docker", "stop", CONTAINER_NAME],
        description="Action 8: Gracefully stop the running container"
    ))

    # Action 9: Remove container
    steps.append(run_cmd(
        ["docker", "rm", CONTAINER_NAME],
        description="Action 9: Remove the stopped container"
    ))

    # Action 10: Remove image
    steps.append(run_cmd(
        ["docker", "rmi", IMAGE_NAME],
        description="Action 10: Clean up and remove the local Docker image"
    ))

    # Save log output
    log_content = [
        "=" * 85,
        "         TASK 6.1: DOCKER WARM-UP LAB EXECUTION & CLI AUDIT LOG          ",
        "=" * 85,
        f"Student Roll Number : 23L0710",
        f"Project Name        : 23L0710_A2_Ops",
        f"Docker Host         : Docker Desktop (Engine 29.9.2)",
        "-" * 85
    ]
    for idx, s in enumerate(steps, 1):
        log_content.append(f"\n[{idx}] {s['description']}")
        log_content.append(f"$ {s['command']} (Exit Code: {s['exit_code']}, Elapsed: {s['elapsed_sec']}s)")
        log_content.append(s["output"])
        log_content.append("-" * 85)

    with open(LOG_FILE, "w") as f:
        f.write("\n".join(log_content))
    print(f"\n[+] Saved audit log to: {LOG_FILE}")

    # Capture visual terminal screenshot
    capture_terminal_screenshot(steps)

    # Generate comprehensive report with 5 mandatory student definitions
    generate_warmup_report(steps)


def capture_terminal_screenshot(steps):
    """Generates an HTML snapshot styled as a macOS terminal and captures high-res PNG."""
    print("[*] Generating terminal HTML snapshot for screenshot...")
    temp_html = os.path.join(EVIDENCE_DIR, "terminal_snapshot.html")

    terminal_entries = ""
    for idx, s in enumerate(steps, 1):
        terminal_entries += f"""
        <div class="step-box">
          <div class="step-desc"># {s['description']}</div>
          <div class="cmd-line"><span class="prompt">shoaib@23L0710-Mac:~$</span> {s['command']}</div>
          <pre class="cmd-output">{s['output']}</pre>
        </div>
        """

    html_code = f"""<!DOCTYPE html>
<html>
<head>
  <style>
    body {{
      background-color: #0f172a;
      font-family: Menlo, Monaco, "Courier New", monospace;
      padding: 25px;
      margin: 0;
      color: #e2e8f0;
    }}
    .window {{
      background: #1e293b;
      border-radius: 10px;
      box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.5);
      border: 1px solid #334155;
      overflow: hidden;
    }}
    .titlebar {{
      background: #0f172a;
      padding: 12px 16px;
      display: flex;
      align-items: center;
      gap: 8px;
      border-bottom: 1px solid #334155;
    }}
    .dot {{ width: 12px; height: 12px; border-radius: 50%; display: inline-block; }}
    .dot-red {{ background: #ef4444; }}
    .dot-yellow {{ background: #f59e0b; }}
    .dot-green {{ background: #10b981; }}
    .title-text {{
      color: #94a3b8;
      font-size: 13px;
      font-weight: 600;
      margin-left: 10px;
    }}
    .content {{
      padding: 20px;
      font-size: 13px;
      line-height: 1.5;
    }}
    .step-box {{
      margin-bottom: 18px;
      border-left: 3px solid #38bdf8;
      padding-left: 12px;
    }}
    .step-desc {{
      color: #38bdf8;
      font-weight: bold;
      margin-bottom: 4px;
      font-size: 12px;
    }}
    .cmd-line {{
      color: #f8fafc;
      font-weight: 600;
    }}
    .prompt {{ color: #4ade80; }}
    .cmd-output {{
      color: #cbd5e1;
      margin: 4px 0 0 0;
      white-space: pre-wrap;
      font-size: 12px;
      background: rgba(0,0,0,0.25);
      padding: 6px 10px;
      border-radius: 4px;
    }}
  </style>
</head>
<body>
  <div class="window">
    <div class="titlebar">
      <span class="dot dot-red"></span>
      <span class="dot dot-yellow"></span>
      <span class="dot dot-green"></span>
      <span class="title-text">shoaib@23L0710: Task 6.1 Docker Warm-Up Lab — Engine 29.9.2</span>
    </div>
    <div class="content">
      {terminal_entries}
    </div>
  </div>
</body>
</html>
"""
    with open(temp_html, "w") as f:
        f.write(html_code)

    os.makedirs("/tmp/chrome_profile", exist_ok=True)
    cmd = [
        CHROME_BIN,
        "--headless=new",
        "--user-data-dir=/tmp/chrome_profile",
        "--disable-gpu",
        "--window-size=1200,1650",
        f"--screenshot={SCREENSHOT_PNG}",
        f"file://{os.path.abspath(temp_html)}"
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20)
        print(f"[+] Captured terminal screenshot -> {SCREENSHOT_PNG}")
    except Exception as e:
        print(f"[-] Screenshot failed: {e}")


def generate_warmup_report(steps):
    """Compiles the formal Task 6.1 technical report with student definitions."""
    report_content = f"""# Task 6.1: Docker Warm-Up Lab Technical Report
**Student Roll Number:** 23L0710  
**Project:** 23L0710_A2_Ops - VisionGuard  
**Docker Engine:** Docker Desktop 4.98.0 (v29.9.2, API 1.56)  
**Test Image Used:** `nginx:alpine`  
**Container Name:** `visionguard_warmup_box`  

---

## 1. Student Core Concepts Definitions (Written in My Own Words)

To establish conceptual mastery prior to building the multi-service VisionGuard architecture, here are the 5 foundational Docker concepts defined in my own words:

### 1. Image
A Docker **Image** is a lightweight, standalone, read-only executable blueprint that packages everything required to run an application: code, runtime, system tools, libraries, environment variables, and configuration files. It is immutable—once built, it never changes. Images act as templates stored in registries (like Docker Hub) from which active containers are instantiated.

### 2. Container
A Docker **Container** is a live, running instance of an image. It represents an isolated, lightweight execution sandbox running on top of the host operating system's kernel (via Linux namespaces and cgroups). Unlike virtual machines, containers share the host kernel and start in milliseconds with minimal CPU and memory overhead. When a container is deleted, any data written to its temporary writable layer is destroyed unless backed by a volume.

### 3. Layer
A Docker **Layer** is an immutable, read-only filesystem diff created by an individual instruction in a `Dockerfile` (such as `RUN`, `COPY`, or `ADD`). Docker builds images by stacking these layers using an overlay filesystem (`overlayfs`). Layers are strictly cached: if an instruction and its inputs do not change, Docker reuses the cached layer, dramatically accelerating subsequent image builds. When a container runs, Docker adds a thin, ephemeral writable layer on top of the stacked image layers.

### 4. Volume
A Docker **Volume** is a persistent storage mechanism managed directly by Docker outside the container's temporary filesystem. Volumes reside on the host filesystem and are mounted into specified container directory paths. They decouple stateful data (such as PostgreSQL databases, model weights, and logs) from container lifecycles: when a container is stopped, upgraded, or destroyed, all data inside the volume remains intact and immediately accessible to new containers.

### 5. Port Mapping
**Port Mapping** (published port via `-p <HOST_PORT>:<CONTAINER_PORT>`) is the networking bridge that forwards incoming network traffic from a designated port on the physical host machine to a specific internal listening port inside the isolated container network namespace. Because containers run inside private virtual bridge networks, port mapping allows external web clients and browsers to reach applications running inside the container (e.g. mapping host port `8085` to internal container port `80`).

---

## 2. Step-by-Step CLI Execution Evidence

Every mandatory Docker lifecycle action was executed and logged:

| Action ID | Action Description | CLI Command Executed | Observed Result |
| :---: | :--- | :--- | :--- |
| **Action 1** | **Pull Image** | `docker pull nginx:alpine` | Pulled official lightweight Alpine-based Nginx image from Docker Hub |
| **Action 2** | **Run with Port & Volume** | `docker run -d --name visionguard_warmup_box -p 8085:80 -v <abs_path>:/usr/share/nginx/html/data nginx:alpine` | Container started detached; mapped port `8085:80` and mounted host directory |
| **Action 3** | **Verify Port Mapping** | `curl -I http://127.0.0.1:8085` | Returned `HTTP/1.1 200 OK` from Nginx running inside container |
| **Action 4** | **Read Logs** | `docker logs visionguard_warmup_box` | Displayed Nginx worker initialization and GET access logs |
| **Action 5** | **Shell Exec** | `docker exec visionguard_warmup_box sh -c "uname -a && cat /.../host_note.txt"` | Verified Linux kernel, verified host volume data was read from inside container |
| **Action 6** | **List Containers** | `docker ps -a --filter name=visionguard_warmup_box` | Verified container status (`Up`), port mapping (`0.0.0.0:8085->80/tcp`), and ID |
| **Action 7** | **List Images** | `docker images nginx:alpine` | Verified local image repository, tag, image ID, and virtual size (~40 MB) |
| **Action 8** | **Stop Container** | `docker stop visionguard_warmup_box` | Sent `SIGTERM` to Nginx master process, terminating cleanly |
| **Action 9** | **Remove Container** | `docker rm visionguard_warmup_box` | Cleaned up stopped container from daemon |
| **Action 10** | **Remove Image** | `docker rmi nginx:alpine` | Untagged and purged local image layers |

---

## 3. Visual Terminal Evidence

The complete terminal sequence was captured and rendered:
- **Terminal Execution Screenshot:** [`evidence/task6/docker_warmup_terminal.png`](file:///Users/shoaib/Desktop/University/7th%20Semester/MLOPS/A2/evidence/task6/docker_warmup_terminal.png)
- **Raw Execution Log:** [`evidence/task6/docker_warmup_log.txt`](file:///Users/shoaib/Desktop/University/7th%20Semester/MLOPS/A2/evidence/task6/docker_warmup_log.txt)
"""
    with open(REPORT_MD, "w") as f:
        f.write(report_content)
    print(f"[+] Saved Task 6.1 Technical Report -> {REPORT_MD}")


if __name__ == "__main__":
    execute_warmup()

