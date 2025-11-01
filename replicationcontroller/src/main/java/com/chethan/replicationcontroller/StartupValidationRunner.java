package com.chethan.replicationcontroller;

import com.chethan.replicationcontroller.config.ClusterConfig;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.CommandLineRunner;
import org.springframework.stereotype.Component;



@Component
public class StartupValidationRunner implements CommandLineRunner {
    private static final Logger logger = LoggerFactory.getLogger(StartupValidationRunner.class);

    @Autowired
    private ClusterConfig clusterConfig;

    @Override
    public void run(String... args) throws Exception {
        logger.info("==================================================");
        logger.info("CONTROLLER STARTING UP");
        logger.info("Managing nodes: {}", clusterConfig.getNodes());
        logger.info("==================================================");
    }
}
