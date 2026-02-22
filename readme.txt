Requirements
- Docker Desktop installed and running
- Virtualization enabled
- All project files in the same folder:
 • server.py
 • client.py
 • centralised.py
 • Dockerfile
 • docker-compose.yml
How to Run
1) Open PowerShell and navigate to your project folder:
cd "C:\Users\Simra\OneDrive - Newcastle University\fedavg\fed learning docker test"
2) Build and start the system:
docker compose up --build
3) Stop the system:
Press Ctrl + C
Then run:
docker compose down

Just copy and paste this, make you life easier
docker compose down
docker compose build --no-cache
docker compose up