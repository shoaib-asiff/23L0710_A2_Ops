# Task 6.1: Docker Warm-Up Lab Technical Report
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
