FROM maven:3.9.11-amazoncorretto-25 AS build

# Set the working directory inside the container
WORKDIR /app

# Copy the Maven project definition file
COPY pom.xml .

# Copy the rest of the source code
COPY src ./src

# Run the Maven package command to compile the code and create the JAR file.
# -DskipTests skips running unit tests during the build for speed.
RUN mvn package -DskipTests


# --- Stage 2: Runtime Stage ---
# Use a lightweight official OpenJDK image for the final runtime environment
FROM openjdk:25-jdk-slim

# Set the working directory
WORKDIR /app

# Copy the JAR file created in the 'build' stage into our new stage
COPY --from=build /app/target/*.jar app.jar

# Expose the port the application will run on (e.g., 8080)
EXPOSE 8080

# The command to run when the container starts.
# This will execute our Spring Boot application.
ENTRYPOINT ["java", "-jar", "app.jar"]