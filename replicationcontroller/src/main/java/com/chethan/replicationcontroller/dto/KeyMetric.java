package com.chethan.replicationcontroller.dto;

import lombok.AllArgsConstructor;
import lombok.Data;

@Data
@AllArgsConstructor
public class KeyMetric {
    private long readCount;
    private long writeCount;
}
