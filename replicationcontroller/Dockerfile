FROM maven:3.9.11-amazoncorretto-25 AS build

WORKDIR /app

COPY pom.xml .

COPY src ./src

RUN mvn package -DskipTests

FROM openjdk:25-jdk-slim

WORKDIR /app

COPY --from=build /app/target/*.jar app.jar

EXPOSE 8080

ENTRYPOINT ["java", "-jar", "app.jar"]