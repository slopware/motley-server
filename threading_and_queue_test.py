import threading
import queue
import time

def worker(q):
    while True:
        item = q.get()
        if item is None:  # None is used as a signal to stop the worker
            break
        # Simulate doing work by sleeping
        print(f"Processing {item}")
        time.sleep(1)
        q.task_done()

def main():
    q = queue.Queue()
    threads = []
    for _ in range(4):  # Create 4 worker threads
        t = threading.Thread(target=worker, args=(q,))
        t.start()
        threads.append(t)
    
    # Enqueue items
    for item in range(20):
        q.put(item)
    
    # Block until all tasks are done
    q.join()
    
    # Stop workers
    for _ in range(4):
        q.put(None)
    for t in threads:
        t.join()

if __name__ == "__main__":
    main()
