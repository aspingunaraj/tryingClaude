# Gunicorn configuration — loaded automatically when gunicorn is started
# from the project directory.

workers     = 1
worker_class = "gthread"   # thread-based: background threads don't block heartbeat
threads     = 4
timeout     = 300          # 5 min — covers long optimize runs
loglevel    = "info"
