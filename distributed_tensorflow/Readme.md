# Distributed tensorflow

Let tensorflow run in muti server.

Suppose we have two servers: server1 and server2 with name hostname1 and hostname2.

Run tf_std_server.py on server1 with index 0, on server2 with index 1.

```bash
sudo docker run --rm -d --network=host -v /mnt/data:/tmp -v $PWD:/root -w /root tensorflow/tensorflow:1.13.1-py3 python ./tf_std_server.py
```

Run estimator_example.py on client.

```bash
sudo docker run --rm --network=host -v /mnt/data:/tmp -v $PWD:/root -w /root tensorflow/tensorflow:1.13.1-py3 python ./estimator_example.py
```

Note that server1, server2 and client all share the same model directory /mnt/data/tfestimator_example.

Can accomplish that through NFS.

```bash
sudo docker run --rm -d --name nfs --privileged -p 2049:2049 -v /tmp:/nfsshare -e SYNC=true -e SHARED_DIRECTORY=/nfsshare itsthenetwork/nfs-server-alpine:12

sudo mount -v 10.11.12.101:/ /some/where/here
sudo umount /some/where/here
```