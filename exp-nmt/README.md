### Run docker images 
```cmd
docker run --rm -p 7080:7080 phrase_transformer:latest
```

### Query to docker model serving 
```cmd
curl -X POST localhost:7080/test-vizh -H "Content-Type: text/xml" --data-binary "path/to/file/sample_input.txt"
```
where `sample_input.txt` is the input text file with content following: 
```
This is a test sample. Each line is one sample for translation
This is a second test sample.
```
