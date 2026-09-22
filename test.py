import requests
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

URL = "http://100.94.254.20:8000/scanned"
PDF_FILE = "Scanned_Answersheet.pdf"
MODEL_ANSWER_ID = "6ab2547f4082b27c89cf4982"

TOTAL_REQUESTS = 10
OUTPUT_FILE = "request_times.txt"

barrier = Barrier(TOTAL_REQUESTS)


def send_request(request_number):
    barrier.wait()

    with open(PDF_FILE, "rb") as f:
        response = requests.post(
            URL,
            files={
                "myFile": (
                    PDF_FILE,
                    f,
                    "application/pdf"
                )
            },
            data={
                "preprocess": "true",
                "model_answer_id": MODEL_ANSWER_ID
            },
            headers={"accept": "*/*"},
            timeout=None
        )

    result = response.json()
    elapsed = result.get("total_elapsed_seconds")

    return request_number, elapsed


if __name__ == "__main__":

    with ThreadPoolExecutor(max_workers=TOTAL_REQUESTS) as executor:
        futures = [
            executor.submit(send_request, i)
            for i in range(1, TOTAL_REQUESTS + 1)
        ]

        results = [future.result() for future in futures]

    # Sort by request number
    results.sort(key=lambda x: x[0])

    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        for request_number, elapsed in results:
            f.write(f"Request {request_number}: {elapsed} seconds\n")

    print(f"Done. Results saved to {OUTPUT_FILE}")