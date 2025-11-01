package com.chethan.projects.replication.dto;
import lombok.Data;
import lombok.AllArgsConstructor;

@Data
@AllArgsConstructor
public class KeyMetric {
    private long readCount;
    private long writeCount;
}
