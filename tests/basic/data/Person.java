package tests.basic.data;

public class Person {
    private String name;
    private int age;
    private Company company;

    public Person(String name, int age, Company company) {
        this.name = name;
        this.age = age;
        this.company = company;
    }

    public String getName() {
        return name;
    }

    public int getAge() {
        return age;
    }

    public Company getCompany() {
        return company;
    }

    @Override
    public String toString() {
        return "Person{name='" + name + "', age=" + age + ", company=" + company + "}";
    }
} 