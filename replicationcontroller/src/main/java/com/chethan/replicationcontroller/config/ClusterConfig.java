package com.chethan.replicationcontroller.config;

import lombok.Data;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.stereotype.Component;

import java.util.List;

@Component
@ConfigurationProperties(prefix = "cluster")
@Data
public class ClusterConfig {

    /**
     * A list of base URLs for all DB Nodes in the cluster.
     * Populated from the 'cluster.nodes' property.
     */
    private List<String> nodes;
}
