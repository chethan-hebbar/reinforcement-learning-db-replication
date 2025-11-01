package com.chethan.replicationcontroller.controller;

import com.chethan.replicationcontroller.dto.ClientReadResponse;
import com.chethan.replicationcontroller.dto.WriteRequest;
import com.chethan.replicationcontroller.service.ReplicationService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/v1/data")
public class DataController {

    @Autowired
    private ReplicationService replicationService;

    @PostMapping
    public ResponseEntity<Void> writeData(@RequestBody WriteRequest writeRequest) {
        replicationService.handleWrite(writeRequest.getKey(), writeRequest.getValue());
        return ResponseEntity.status(HttpStatus.CREATED).build();
    }

    @GetMapping("/{key}")
    public ResponseEntity<ClientReadResponse> getData(@PathVariable String key) {
        ClientReadResponse response = replicationService.handleRead(key);
        return ResponseEntity.ok(response);
    }
}
