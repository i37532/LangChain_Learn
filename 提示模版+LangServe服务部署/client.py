from langserve import RemoteRunnable

if __name__ == '__main__':
    client = RemoteRunnable('http://127.0.0.1:8000/chaindemo/')
    print(client.invoke({'language':'Japanese','text':'识时务者为俊杰'}))