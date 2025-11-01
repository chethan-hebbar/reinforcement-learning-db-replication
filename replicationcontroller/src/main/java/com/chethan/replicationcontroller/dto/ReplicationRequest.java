package com.chethan.replicationcontroller.dto;

import lombok.AllArgsConstructor;
import lombok.Data;

@Data
@AllArgsConstructor
public class ReplicationRequest {
    private String key;
    private String value;
}