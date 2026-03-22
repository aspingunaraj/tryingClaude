# Gunicorn configuration — loaded automatically when gunicorn is started
# from the project directory.

workers         = 1
worker_class    = "gthread"   # thread-based: background threads don't block heartbeat
threads         = 4
timeout         = 300          # 5 min — covers long optimize runs
graceful_timeout = 120         # give background jobs 2 min to finish on shutdown/deploy
loglevel        = "info"

def on_starting(server):
    server.log.info(
        "Gunicorn starting — worker_class=%s  timeout=%s  graceful_timeout=%s",
        server.cfg.worker_class_str,
        server.cfg.timeout,
        server.cfg.graceful_timeout,
    )
