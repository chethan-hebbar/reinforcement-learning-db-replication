package com.chethan.replicationcontroller.dto;

import lombok.Data;

@Data
public class WriteRequest {
    private String key;
    private String value;
}