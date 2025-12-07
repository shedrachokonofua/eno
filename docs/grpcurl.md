# gRPC Inspection

```bash
task lute:grpcurl -- list                           # List services
task lute:grpcurl -- describe lute.AlbumService     # Describe service
task lute:grpcurl -- lute.EventService/GetMonitor   # Call method
```

Shows streams, subscribers, cursors. Useful for debugging extraction without code changes.
