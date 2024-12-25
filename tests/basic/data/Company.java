package tests.basic.data;

public class Company {
    private String name;
    private String industry;
    private Address address;

    public Company(String name, String industry) {
        this.name = name;
        this.industry = industry;
    }

    public String getName() {
        return name;
    }

    public String getIndustry() {
        return industry;
    }

    public Address getAddress() {
        return address;
    }

    @Override
    public String toString() {
        return "Company{name='" + name + "', industry='" + industry + "'}";
    }
} 