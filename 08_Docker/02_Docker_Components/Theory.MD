# **Docker Engine**
Docker Engine is the core component of the Docker Platform, responsible for creating, running and managing Docker containers, it serves as the runtime that powers Docker's containerization capabilities. 

[Image of the docker]
Docker Daemon is the main thing. 
First we give some commands to the CLI and then CLI tells to the REST API and then that API tells to the Daemon for working... So, Docker Daemon or server is the most important thing in this above illustration.

We can send: images, containers, network and data volumes from the CLI to the docker daemon.

## **Components of Docker Engine**
- **Docker Daemon(dockerd):**
    - **Function:** The docker daemon is the background service running on the host machine. It manages docker objects such as: images, containers, networks and volumes.
    - **Interaction:** It listens for Docker API requests and processes them, handling container lifecycle operations (start, stop, restart, etc).
- **Docker CLI(docker):**
    - **Function:** The docker command line interface(CLI) is the tool that users interact with to communicate with the Docker daemon
    - **Usage:** Users run docker commands through the CLI to perform tasks like building images, running containers, and managing Docker resources.
- **REST API:**
    - **Function:** The docker REST API allows communication between the Docker CLI and the docker daemon. It also enables programmatic interaction with Docker.
    - **Usage:** Developers can use the API to automate Docker operations or integrate Docker functionality into their applications.

## **Docker Image**
A docker image is a lightweight, stand-alone, and executable software package that includes everything needed to run a piece of software, such as the code, runtime, libraries, environment variables, and configuration files. Images are used to create Docker containers which are instances of these images.

**Components of a Docker Image**
- **Base Image:** The starting point for building an image. It could be a minimal OS image like alpine, a full-fledged OS like: Ubuntu or even another application image like: python or node.
- **Application Code:** The actual code and files necessary for the application to run.
- **Dependencies:** Libraries, Frameworks and packages required by the application.
- **Metadata:** Information about the image, such as environment variables, labels and exposed ports.

**Docker Image LifeCycle**
- **Creation:** Images are created using docker build command, which processes the instructions in a Dockerfile to create the image layers.
- **Storage:** Images are stored locally on the host machine. They can also be pushed to and pulled from Docker registries like: DockerHub, AWS ECR, or Google Container Registry.
- **Distribution:** Images can be shared by pushing them to a Docker registry, allowing others to pull and use the same image.
- **Execution:** Images are executed by running containers, which are instances of these images

## **DockerFile**